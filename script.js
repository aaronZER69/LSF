/* ============================================================
   Signe-moi — apprendre l'alphabet manuel avec la webcam
   TP3 partie 2 — JavaScript (MMI)

   Principe :
   1. On demande l'accès à la webcam (getUserMedia) dès l'arrivée.
   2. ml5.handPose détecte 21 points (landmarks) de la main.
   3. Une fonction par lettre analyse la géométrie de ces points
      (doigt tendu / plié / replié, distances entre bouts de doigts).
   4. Si la lettre demandée est reconnue pendant HOLD_MS ms,
      elle est validée et on passe à la suivante.

   Plan du fichier :
   A. Options et paramètres
   B. Références DOM et état
   C. Reconnaissance des lettres (analyse des points de la main)
   D. Fonctions d'affichage (carte, progression, caméra)
   E. Logique de jeu (validation, navigation)
   F. Callbacks (webcam, ml5)
   G. Écouteurs d'événements et initialisation
   ============================================================ */

"use strict";

/* ============================================================
   A. OPTIONS ET PARAMÈTRES
   ============================================================ */

const CONFIG = {
  /* MODE TEST SANS WEBCAM : true = pas de caméra, la touche Entrée valide
     la lettre en cours. À repasser sur false pour le rendu ! */
  DEBUG_NO_CAM: true,

  /* Dossier contenant les images de démonstration (A.png, B.png, …) */
  IMAGES_DIR: "images/",

  /* Durée pendant laquelle le signe doit être tenu pour être validé (ms) */
  HOLD_MS: 900,
  /* Tolérance : on ne remet pas le chrono à zéro si le signe saute
     pendant moins que ça (ms). Évite les "tremblements" de détection. */
  GRACE_MS: 250,
  /* Délai avant d'afficher la lettre suivante après une réussite (ms) */
  NEXT_DELAY_MS: 900,
  /* Durée du retournement de carte (doit correspondre au CSS : .7s) */
  FLIP_MS: 700,

  /* Options passées au modèle ml5 */
  MODEL_OPTIONS: { maxHands: 1, flipped: false },

  /* Contraintes de la caméra */
  VIDEO_CONSTRAINTS: { video: { width: 640, height: 480, facingMode: "user" }, audio: false },

  /* Seuils de géométrie (valeurs relatives à la taille de la paume).
     À ajuster si la reconnaissance est trop stricte ou trop souple. */
  T: {
    EXTENDED: 1.65,   // rapport (bout du doigt→poignet)/(base du doigt→poignet) : au-dessus = tendu
    CURLED: 1.15,     // en dessous = replié ; entre les deux = plié (courbé)
    THUMB_OUT: 0.9,   // pouce écarté de la main (distance bout du pouce → base de l'index)
    TOUCH: 0.4,       // deux bouts de doigts "se touchent"
    TOGETHER: 0.4     // deux doigts collés (U) / écartés (V)
  },

  /* Couleurs du squelette dessiné sur la vidéo */
  COLORS: { bones: "#ffc93c", joints: "#ffffff" }
};

/* Alphabet travaillé (lettres statiques uniquement : J et Z demandent
   un mouvement, K/M/N/P/Q/R/T/E/G/H trop ambiguës en 2D). */
const LETTERS = [
  { letter: "A", hint: "Poing fermé, pouce collé sur le côté du poing." },
  { letter: "B", hint: "Quatre doigts tendus et serrés, pouce replié sur la paume." },
  { letter: "C", hint: "Main en forme de C, doigts et pouce arrondis." },
  { letter: "D", hint: "Index tendu, les autres doigts rejoignent le pouce." },
  { letter: "F", hint: "Pouce et index se touchent, les trois autres doigts sont tendus." },
  { letter: "I", hint: "Poing fermé, seul l'auriculaire (petit doigt) est tendu." },
  { letter: "L", hint: "Index et pouce tendus en L, autres doigts repliés." },
  { letter: "O", hint: "Tous les doigts arrondis rejoignent le pouce pour former un O." },
  { letter: "S", hint: "Poing fermé, pouce passé en travers devant les doigts." },
  { letter: "U", hint: "Index et majeur tendus et collés, autres doigts repliés." },
  { letter: "V", hint: "Index et majeur tendus et écartés en V." },
  { letter: "W", hint: "Index, majeur et annulaire tendus et écartés." },
  { letter: "X", hint: "Index courbé en crochet, autres doigts repliés." },
  { letter: "Y", hint: "Pouce et auriculaire tendus, les autres doigts repliés." }
];

/* ============================================================
   B. RÉFÉRENCES DOM ET ÉTAT
   ============================================================ */

const el = {
  gate: document.getElementById("gate"),
  gateError: document.getElementById("gate-error"),
  btnAllow: document.getElementById("btn-allow"),
  btnStart: document.getElementById("btn-start"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  status: document.getElementById("status"),
  card: document.getElementById("card"),
  cardFrontSub: document.getElementById("card-front-sub"),
  cardLetter: document.getElementById("card-letter"),
  cardHint: document.getElementById("card-hint"),
  cardSign: document.getElementById("card-sign"),
  holdBar: document.getElementById("hold-bar"),
  feedback: document.getElementById("feedback"),
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  cam: document.querySelector(".cam"),
  detected: document.getElementById("detected"),
  progress: document.getElementById("progress")
};
const ctx = el.overlay.getContext("2d");

const state = {
  model: null,            // modèle ml5 handPose
  modelReady: false,
  running: false,         // true entre Start et Stop
  stream: null,           // MediaStream de la webcam
  hands: [],              // dernières mains détectées
  index: 0,               // lettre en cours
  validated: new Set(),   // indices des lettres réussies
  holdStart: null,        // début du maintien du bon signe
  lastMatch: 0,           // dernier instant où le bon signe a été vu
  locked: false           // true pendant le retournement / après réussite
};

/* ============================================================
   C. RECONNAISSANCE DES LETTRES
   ============================================================ */

/** Distance entre deux points {x, y}. */
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Calcule les caractéristiques d'une main à partir des 21 points ml5.
 * Toutes les distances sont divisées par la taille de la paume (poignet →
 * base du majeur), ce qui rend l'analyse indépendante de la distance à la caméra.
 */
function computeFeatures(hand) {
  const k = {};
  hand.keypoints.forEach(p => { k[p.name] = p; });

  const palm = dist(k.wrist, k.middle_finger_mcp) || 1;
  const d = (a, b) => dist(k[a], k[b]) / palm;

  /* État de chaque doigt : "e" tendu, "b" plié (courbé), "c" replié */
  const fingerState = name => {
    const ratio = dist(k[name + "_tip"], k.wrist) / dist(k[name + "_mcp"], k.wrist);
    if (ratio > CONFIG.T.EXTENDED) return "e";
    if (ratio < CONFIG.T.CURLED) return "c";
    return "b";
  };

  return {
    index: fingerState("index_finger"),
    middle: fingerState("middle_finger"),
    ring: fingerState("ring_finger"),
    pinky: fingerState("pinky_finger"),
    thumbOut: d("thumb_tip", "index_finger_mcp") > CONFIG.T.THUMB_OUT,
    thumbOnPalm: d("thumb_tip", "ring_finger_mcp") < 0.7,
    thumbIndexTip: d("thumb_tip", "index_finger_tip"),
    thumbMiddleTip: d("thumb_tip", "middle_finger_tip"),
    thumbMiddlePip: d("thumb_tip", "middle_finger_pip"),
    indexMiddleTips: d("index_finger_tip", "middle_finger_tip")
  };
}

/* Un détecteur par lettre : reçoit les caractéristiques, renvoie true/false. */
const DETECTORS = {
  A: f => [f.index, f.middle, f.ring, f.pinky].every(s => s === "c") && f.thumbMiddlePip > 0.55,
  B: f => [f.index, f.middle, f.ring, f.pinky].every(s => s === "e") && f.thumbOnPalm,
  C: f => [f.index, f.middle, f.ring, f.pinky].every(s => s === "b") &&
          f.thumbIndexTip > 0.5 && f.thumbIndexTip < 1.3,
  D: f => f.index === "e" && f.middle !== "e" && f.ring !== "e" && f.pinky !== "e" &&
          f.thumbMiddleTip < 0.55,
  F: f => f.index !== "e" && f.middle === "e" && f.ring === "e" && f.pinky === "e" &&
          f.thumbIndexTip < CONFIG.T.TOUCH,
  I: f => f.pinky === "e" && [f.index, f.middle, f.ring].every(s => s === "c") && !f.thumbOut,
  L: f => f.index === "e" && f.middle === "c" && f.ring === "c" && f.pinky === "c" && f.thumbOut,
  O: f => f.index === "b" && f.middle === "b" && f.ring !== "e" && f.pinky !== "e" &&
          f.thumbIndexTip < 0.45 && f.thumbMiddleTip < 0.6,
  S: f => [f.index, f.middle, f.ring, f.pinky].every(s => s === "c") && f.thumbMiddlePip <= 0.55,
  U: f => f.index === "e" && f.middle === "e" && f.ring === "c" && f.pinky === "c" &&
          f.indexMiddleTips < CONFIG.T.TOGETHER,
  V: f => f.index === "e" && f.middle === "e" && f.ring === "c" && f.pinky === "c" &&
          f.indexMiddleTips >= CONFIG.T.TOGETHER,
  W: f => f.index === "e" && f.middle === "e" && f.ring === "e" && f.pinky === "c",
  X: f => f.index === "b" && f.middle === "c" && f.ring === "c" && f.pinky === "c",
  Y: f => f.pinky === "e" && f.thumbOut && [f.index, f.middle, f.ring].every(s => s === "c")
};

/** Renvoie la lettre reconnue pour une main, ou null. */
function classify(hand) {
  const features = computeFeatures(hand);
  const found = LETTERS.find(l => DETECTORS[l.letter](features));
  return found ? found.letter : null;
}

/* ============================================================
   D. AFFICHAGE
   ============================================================ */

/** Retourne la carte pour révéler la lettre courante. */
function showCardForCurrentLetter() {
  state.locked = true;
  el.card.classList.remove("is-flipped", "is-success");   // face cachée
  setTimeout(() => {
    const current = LETTERS[state.index];
    el.cardLetter.textContent = current.letter;
    el.cardHint.textContent = current.hint;
    el.cardSign.src = CONFIG.IMAGES_DIR + current.letter + ".png";   // démonstration du signe
    el.cardSign.alt = "Démonstration du signe de la lettre " + current.letter;
    el.card.classList.add("is-flipped");                   // révèle la lettre
    setTimeout(() => { state.locked = false; }, CONFIG.FLIP_MS);
  }, CONFIG.FLIP_MS / 2);
}

/** Reconstruit la rangée de progression (une pastille par lettre). */
function renderProgress() {
  el.progress.innerHTML = "";
  LETTERS.forEach((l, i) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = l.letter;
    btn.disabled = !state.validated.has(i) && i !== state.index;
    btn.classList.toggle("is-done", state.validated.has(i));
    btn.classList.toggle("is-current", i === state.index);
    btn.setAttribute("aria-label", "Lettre " + l.letter + (state.validated.has(i) ? " réussie" : ""));
    btn.addEventListener("click", () => goToLetter(i));
    li.appendChild(btn);
    el.progress.appendChild(li);
  });
}

/** Met à jour l'état des boutons précédent / suivant. */
function updateNavButtons() {
  el.btnPrev.disabled = !state.running || state.index === 0;
  el.btnNext.disabled = !state.running || !state.validated.has(state.index) ||
                        state.index >= LETTERS.length - 1;
}

function setHold(ratio) {
  el.holdBar.style.width = Math.round(Math.min(ratio, 1) * 100) + "%";
}

function setFeedback(text, kind) {
  el.feedback.textContent = text || "\u00a0";
  el.feedback.className = "feedback" + (kind ? " " + kind : "");
}

/** Dessine le squelette de la main sur le canvas (même repère que la vidéo). */
function drawHands() {
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  state.hands.forEach(hand => {
    const k = {};
    hand.keypoints.forEach(p => { k[p.name] = p; });

    /* Segments : poignet → base de chaque doigt, puis le long de chaque doigt */
    ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky_finger"].forEach(f => {
      const chain = f === "thumb"
        ? ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]
        : ["wrist", f + "_mcp", f + "_pip", f + "_dip", f + "_tip"];
      ctx.beginPath();
      chain.forEach((n, i) => (i ? ctx.lineTo(k[n].x, k[n].y) : ctx.moveTo(k[n].x, k[n].y)));
      ctx.strokeStyle = CONFIG.COLORS.bones;
      ctx.lineWidth = 3;
      ctx.stroke();
    });
    hand.keypoints.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = CONFIG.COLORS.joints;
      ctx.fill();
    });
  });
}

/* ============================================================
   E. LOGIQUE DE JEU
   ============================================================ */

/** Appelée à chaque nouvelle détection : compare le signe à la lettre cible. */
function evaluateSign() {
  if (!state.running || state.locked) return;

  const now = performance.now();
  const hand = state.hands[0];
  const seen = hand ? classify(hand) : null;
  const target = LETTERS[state.index].letter;
  el.detected.textContent = seen || (hand ? "?" : "—");

  if (seen === target) {
    state.lastMatch = now;
    if (state.holdStart === null) state.holdStart = now;
    const ratio = (now - state.holdStart) / CONFIG.HOLD_MS;
    setHold(ratio);
    setFeedback("Parfait, ne bouge plus…", "ok");
    if (ratio >= 1) onLetterSuccess();
  } else {
    /* Tolérance : on garde le chrono si l'interruption est très courte */
    if (state.holdStart !== null && now - state.lastMatch > CONFIG.GRACE_MS) {
      state.holdStart = null;
      setHold(0);
    }
    if (state.holdStart === null) {
      setFeedback(hand ? "Pas tout à fait… observe bien la carte." : "Montre ta main à la caméra.", hand ? "ko" : "");
    }
  }
}

/** Le signe a été tenu assez longtemps : on valide et on enchaîne. */
function onLetterSuccess() {
  state.locked = true;
  state.holdStart = null;
  state.validated.add(state.index);
  el.card.classList.add("is-success");
  setHold(1);
  setFeedback("Bravo, c'est un " + LETTERS[state.index].letter + " !", "ok");
  renderProgress();
  updateNavButtons();

  setTimeout(() => {
    if (state.index < LETTERS.length - 1) {
      goToLetter(state.index + 1);
    } else {
      setFeedback("Alphabet terminé, félicitations !", "ok");
      el.cardFrontSub.textContent = "Tu connais toutes les lettres";
    }
  }, CONFIG.NEXT_DELAY_MS);
}

/** Affiche la lettre i (autorisé seulement si réussie ou en cours). */
function goToLetter(i) {
  if (i < 0 || i >= LETTERS.length) return;
  if (i > state.index && !state.validated.has(state.index) && i !== state.index) return;
  state.index = i;
  state.holdStart = null;
  setHold(0);
  setFeedback("");
  renderProgress();
  updateNavButtons();
  showCardForCurrentLetter();
}

/* ============================================================
   F. CALLBACKS : WEBCAM ET ML5
   ============================================================ */

/** Demande la permission au chargement, puis libère la caméra. */
async function onAllowClick() {
  if (CONFIG.DEBUG_NO_CAM) { el.gate.classList.add("is-hidden"); return; }
  el.gateError.hidden = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia(CONFIG.VIDEO_CONSTRAINTS);
    stream.getTracks().forEach(t => t.stop());   // l'accès est accordé ; on rallume au clic sur Start
    el.gate.classList.add("is-hidden");
  } catch (err) {
    el.gateError.hidden = false;
    el.gateError.textContent = errorMessage(err);
  }
}

function errorMessage(err) {
  switch (err.name) {
    case "NotAllowedError": return "Accès refusé. Autorise la caméra dans la barre d'adresse du navigateur puis réessaie.";
    case "NotFoundError": return "Aucune caméra détectée sur cet appareil.";
    case "NotReadableError": return "La caméra est utilisée par une autre application.";
    default: return window.isSecureContext
      ? "Impossible d'accéder à la caméra (" + err.name + ")."
      : "La caméra n'est accessible qu'en HTTPS ou sur localhost.";
  }
}

/** Callback ml5 : le modèle est chargé. */
function onModelReady() {
  state.modelReady = true;
  el.btnStart.disabled = false;
  el.btnStart.textContent = "Start";
  el.status.textContent = "Prêt. Appuie sur Start pour allumer la caméra.";
}

/** Callback ml5 : appelé à chaque détection de mains. */
function onHandsDetected(results) {
  state.hands = results;
  evaluateSign();
}

/** Boucle de dessin, indépendante de la fréquence de détection. */
function drawLoop() {
  if (!state.running) return;
  drawHands();
  requestAnimationFrame(drawLoop);
}

async function startCamera() {
  if (CONFIG.DEBUG_NO_CAM) {
    /* Mode test : pas de flux vidéo, on simule juste la zone caméra */
    el.overlay.width = 640;
    el.overlay.height = 480;
    el.cam.classList.add("is-on");
  } else {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia(CONFIG.VIDEO_CONSTRAINTS);
    } catch (err) {
      el.status.textContent = errorMessage(err);
      return;
    }
    el.video.srcObject = state.stream;
    await el.video.play();
    el.overlay.width = el.video.videoWidth;     // le canvas utilise le repère de la vidéo
    el.overlay.height = el.video.videoHeight;
    el.cam.classList.add("is-on");
  }

  state.running = true;
  state.index = 0;
  state.validated.clear();
  state.hands = [];
  el.btnStart.textContent = "Arrêter";
  el.status.textContent = "Caméra active.";

  renderProgress();
  updateNavButtons();
  showCardForCurrentLetter();
  if (!CONFIG.DEBUG_NO_CAM) state.model.detectStart(el.video, onHandsDetected);
  requestAnimationFrame(drawLoop);
}

function stopCamera() {
  state.running = false;
  if (!CONFIG.DEBUG_NO_CAM) state.model.detectStop();
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());
  el.video.srcObject = null;
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.cam.classList.remove("is-on");
  el.card.classList.remove("is-flipped", "is-success");
  el.btnStart.textContent = "Start";
  el.status.textContent = "Caméra arrêtée.";
  el.detected.textContent = "—";
  setHold(0);
  setFeedback("");
  updateNavButtons();
}

/* ============================================================
   G. ÉCOUTEURS ET INITIALISATION
   ============================================================ */

el.btnAllow.addEventListener("click", onAllowClick);
el.btnStart.addEventListener("click", () => (state.running ? stopCamera() : startCamera()));
el.btnPrev.addEventListener("click", () => goToLetter(state.index - 1));
el.btnNext.addEventListener("click", () => goToLetter(state.index + 1));

/* Mode test sans webcam : Entrée = valider la lettre en cours */
if (CONFIG.DEBUG_NO_CAM) {
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && state.running && !state.locked) onLetterSuccess();
  });
}

/* Chargement du modèle dès l'arrivée, pour que Start soit instantané */
state.model = ml5.handPose(CONFIG.MODEL_OPTIONS, onModelReady);
renderProgress();