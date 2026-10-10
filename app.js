// Forma Certa — protótipo: análise de agachamento pela câmera, no próprio celular.
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODELO_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const CHAVE_HIST = "formacerta_historico";

// Pontos do corpo (numeração do MediaPipe)
const P = { nariz: 0, ombroE: 11, ombroD: 12, quadrilE: 23, quadrilD: 24, joelhoE: 25, joelhoD: 26, tornozeloE: 27, tornozeloD: 28 };
const LIGACOES = [[11, 12], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [11, 13], [13, 15], [12, 14], [14, 16], [27, 31], [28, 32]];

// Limites do agachamento (ajustáveis depois dos testes)
const LIM = {
  emPe: 160,           // ângulo do joelho considerado "em pé"
  inicioDescida: 140,  // abaixo disso começou a descer
  minimoRep: 135,      // precisa passar disso para contar repetição
  profundidade: -0.10, // quadril até ~nível do joelho (em comprimentos de coxa)
  inclinacaoTronco: 55,// graus de inclinação do tronco (visto de lado)
  joelhosParaDentro: 0.75, // distância joelhos / distância tornozelos (visto de frente)
  perdaVelocidadeAlerta: 0.30,
};

const $ = (id) => document.getElementById(id);
const els = {
  telas: { inicio: $("tela-inicio"), treino: $("tela-treino"), resumo: $("tela-resumo") },
  peso: $("peso"), camera: $("camera"), voz: $("voz"),
  video: $("video"), canvas: $("desenho"),
  reps: $("reps"), status: $("status"), vista: $("vista"), alerta: $("alerta"),
  angulo: $("angulo"), subida: $("subida"), resumo: $("resumo"), historico: $("historico"),
};
const ctx = els.canvas.getContext("2d");

let detector = null;
let carregandoDetector = null;
let stream = null;
let wakeLock = null;
let rodando = false;
let ultimoTempoVideo = -1;
let serie = null;
let timerAlerta = null;

// ---------- utilidades ----------
const fmt = (n, casas = 1) => n.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
const arred25 = (kg) => Math.max(0, Math.round(kg / 2.5) * 2.5);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function angulo(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y }, v2 = { x: c.x - b.x, y: c.y - b.y };
  const cos = (v1.x * v2.x + v1.y * v2.y) / (Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1);
  return Math.acos(Math.min(1, Math.max(-1, cos))) * 180 / Math.PI;
}
const meio = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

function mostrarTela(nome) {
  Object.values(els.telas).forEach((t) => t.classList.remove("ativa"));
  els.telas[nome].classList.add("ativa");
  window.scrollTo(0, 0);
}

let ultimaFala = "";
let ultimaFalaEm = 0;
function falar(texto, prioridade = false) {
  if (!els.voz.checked || !("speechSynthesis" in window)) return;
  const agora = performance.now();
  if (!prioridade && texto === ultimaFala && agora - ultimaFalaEm < 2500) return;
  ultimaFala = texto; ultimaFalaEm = agora;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(texto);
  u.lang = "pt-BR"; u.rate = 1.1;
  speechSynthesis.speak(u);
}

function mostrarAlerta(texto, bom = false) {
  els.alerta.textContent = texto;
  els.alerta.classList.toggle("bom", bom);
  els.alerta.hidden = false;
  clearTimeout(timerAlerta);
  timerAlerta = setTimeout(() => (els.alerta.hidden = true), 1600);
}

function definirStatus(texto, classe = "") {
  els.status.textContent = texto;
  els.status.className = "status " + classe;
}

// ---------- detector ----------
function carregarDetector() {
  if (carregandoDetector) return carregandoDetector;
  carregandoDetector = (async () => {
    const arquivos = await FilesetResolver.forVisionTasks(WASM_URL);
    const opcoes = (delegate) => ({
      baseOptions: { modelAssetPath: MODELO_URL, delegate },
      runningMode: "VIDEO", numPoses: 1,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try {
      detector = await PoseLandmarker.createFromOptions(arquivos, opcoes("GPU"));
    } catch (e) {
      console.warn("GPU indisponível, usando CPU", e);
      detector = await PoseLandmarker.createFromOptions(arquivos, opcoes("CPU"));
    }
    return detector;
  })();
  carregandoDetector.catch(() => (carregandoDetector = null));
  return carregandoDetector;
}

// ---------- série ----------
function novaSerie() {
  return {
    peso: parseFloat(String(els.peso.value).replace(",", ".")) || 0,
    inicio: Date.now(),
    estado: "aguardando", // aguardando -> em_pe -> descendo -> subindo -> em_pe ...
    paradoDesde: 0,
    anguloSuave: null,
    vistaSuave: null,
    coxa: 0,
    quadrilPeY: null,
    rep: null,
    reps: [],
    alertouFadiga: false,
  };
}

function novaRep(agora) {
  return { inicio: agora, anguloMin: 180, tFundo: agora, quadrilMaxY: -Infinity, profundidadeMax: -Infinity,
    inclinacaoMax: 0, razaoJoelhosMin: Infinity };
}

// Interpreta um quadro da câmera
function analisar(pontos, W, H, agora) {
  const px = (i) => ({ x: pontos[i].x * W, y: pontos[i].y * H, v: pontos[i].visibility ?? 1 });
  const s = serie;

  const lado = (px(P.quadrilE).v + px(P.joelhoE).v + px(P.tornozeloE).v) >= (px(P.quadrilD).v + px(P.joelhoD).v + px(P.tornozeloD).v) ? "E" : "D";
  const q = px(P["quadril" + lado]), j = px(P["joelho" + lado]), t = px(P["tornozelo" + lado]), o = px(P["ombro" + lado]);

  // Corpo inteiro visível?
  const visivel = q.v > 0.5 && j.v > 0.5 && t.v > 0.5 && pontos[P["tornozelo" + lado]].y < 1.02 && pontos[P.nariz].y > -0.05;
  if (!visivel) {
    definirStatus("Mostre o corpo inteiro", "atencao");
    els.vista.textContent = "Afaste o celular ou incline-o";
    return { lado, falhas: [] };
  }

  // De frente ou de lado? (largura dos ombros comparada ao tronco)
  const oE = px(P.ombroE), oD = px(P.ombroD), qE = px(P.quadrilE), qD = px(P.quadrilD);
  const tronco = dist(meio(oE, oD), meio(qE, qD)) || 1;
  const razaoVista = Math.abs(oE.x - oD.x) / tronco;
  // Decidido só enquanto a pessoa está parada em pé; depois fica fixo até o fim da série
  // (durante o agachamento a proporção muda e confundiria a leitura).
  if (s.estado === "aguardando") s.vistaSuave = s.vistaSuave == null ? razaoVista : s.vistaSuave * 0.8 + razaoVista * 0.2;
  const vista = s.vistaSuave > 0.45 ? "frente" : s.vistaSuave < 0.32 ? "lado" : "diagonal";
  els.vista.textContent = vista === "frente" ? "Visto de frente · checando joelhos"
    : vista === "lado" ? "Visto de lado · checando profundidade e tronco"
    : "Na diagonal · prefira bem de lado ou de frente";

  const coxa = dist(q, j) || 1;
  const quadrilY = vista === "frente" ? (qE.y + qD.y) / 2 : q.y;
  const joelhoY = vista === "frente" ? (px(P.joelhoE).y + px(P.joelhoD).y) / 2 : j.y;

  // Ângulo do joelho. De frente o joelho dobra "para dentro da imagem" e o ângulo quase não muda,
  // então estimamos pela descida do quadril (uma coxa inteira de descida ≈ agachamento completo).
  let ang = angulo(q, j, t);
  if (vista === "frente" && s.quadrilPeY != null) {
    const descida = Math.max(0, (quadrilY - s.quadrilPeY) / s.coxa);
    ang = Math.max(40, 180 - descida * 135);
  }
  s.anguloSuave = s.anguloSuave == null ? ang : s.anguloSuave * 0.5 + ang * 0.5;
  const a = s.anguloSuave;
  els.angulo.textContent = Math.round(a) + "°";

  const profundidade = (quadrilY - joelhoY) / (s.coxa || coxa); // 0 = paralelo; positivo = abaixo do joelho
  const inclinacao = Math.atan2(Math.abs(o.x - q.x), Math.abs(q.y - o.y)) * 180 / Math.PI;
  const distTorn = Math.abs(px(P.tornozeloE).x - px(P.tornozeloD).x);
  const razaoJoelhos = distTorn > coxa * 0.25 ? Math.abs(px(P.joelhoE).x - px(P.joelhoD).x) / distTorn : Infinity;

  const falhasAoVivo = [];

  switch (s.estado) {
    case "aguardando":
      // Precisa ficar parado em pé ~1 s para calibrar
      if (a > LIM.emPe) {
        if (!s.paradoDesde) s.paradoDesde = agora;
        definirStatus("Fique parado em pé…", "atencao");
        if (agora - s.paradoDesde > 1000) {
          s.coxa = vista === "frente" ? (dist(qE, px(P.joelhoE)) + dist(qD, px(P.joelhoD))) / 2 : coxa;
          s.quadrilPeY = quadrilY;
          s.estado = "em_pe";
          definirStatus("Pronto! Pode começar", "ok");
          falar("Pronto. Pode começar.", true);
        }
      } else {
        s.paradoDesde = 0;
        definirStatus("Fique em pé para calibrar", "atencao");
      }
      break;

    case "em_pe":
      if (a < LIM.inicioDescida) {
        s.rep = novaRep(agora);
        s.estado = "descendo";
        definirStatus("Descendo", "");
      }
      break;

    case "descendo":
    case "subindo": {
      const r = s.rep;
      if (a < r.anguloMin) { r.anguloMin = a; r.tFundo = agora; r.quadrilMaxY = quadrilY; }
      r.profundidadeMax = Math.max(r.profundidadeMax, profundidade);
      if (vista !== "frente") r.inclinacaoMax = Math.max(r.inclinacaoMax, inclinacao);
      if (vista === "frente") r.razaoJoelhosMin = Math.min(r.razaoJoelhosMin, razaoJoelhos);
      r.vista = vista;

      if (vista === "frente" && razaoJoelhos < LIM.joelhosParaDentro) falhasAoVivo.push("joelhos");
      if (vista !== "frente" && inclinacao > LIM.inclinacaoTronco) falhasAoVivo.push("tronco");

      if (s.estado === "descendo" && a > r.anguloMin + 12) {
        s.estado = "subindo";
        definirStatus("Subindo", "");
      } else if (s.estado === "subindo" && a < r.anguloMin + 5) {
        s.estado = "descendo"; // voltou a descer
      }

      if (a > LIM.emPe - 5) {
        if (r.anguloMin < LIM.minimoRep) concluirRep(r, agora, quadrilY);
        else { mostrarAlerta("Desça mais"); falar("Desça mais"); }
        s.estado = "em_pe";
        s.rep = null;
      }
      break;
    }
  }
  return { lado, vista, falhas: falhasAoVivo };
}

function concluirRep(r, agora, quadrilTopoY) {
  const s = serie;
  const tempoSubida = Math.max(0.15, (agora - r.tFundo) / 1000);
  const deslocamento = Math.max(0, (r.quadrilMaxY - quadrilTopoY) / (s.coxa || 1)); // em coxas
  const velocidade = deslocamento / tempoSubida;

  const falhas = [];
  if (r.profundidadeMax < LIM.profundidade) falhas.push("Faltou profundidade");
  if (r.vista !== "frente" && r.inclinacaoMax > LIM.inclinacaoTronco) falhas.push("Tronco inclinado demais");
  if (r.vista === "frente" && r.razaoJoelhosMin < LIM.joelhosParaDentro) falhas.push("Joelhos para dentro");

  const rep = { n: s.reps.length + 1, tempoSubida, velocidade, anguloMin: r.anguloMin, profundidade: r.profundidadeMax,
    inclinacao: r.inclinacaoMax, falhas, vista: r.vista };
  s.reps.push(rep);

  els.reps.textContent = rep.n;
  els.subida.textContent = fmt(tempoSubida) + " s";

  if (falhas.length) {
    mostrarAlerta(falhas[0]);
    const dica = { "Faltou profundidade": "desça mais", "Tronco inclinado demais": "peito para cima", "Joelhos para dentro": "joelhos para fora" }[falhas[0]];
    falar(`${rep.n}. ${dica}`, true);
  } else {
    mostrarAlerta(`${rep.n} ✔`, true);
    falar(`${rep.n}`, true);
  }

  // Fadiga: velocidade caindo em relação às melhores repetições do início
  const melhor = Math.max(...s.reps.slice(0, 3).map((x) => x.velocidade));
  if (s.reps.length >= 3 && melhor > 0 && !s.alertouFadiga) {
    const perda = 1 - velocidade / melhor;
    if (perda > LIM.perdaVelocidadeAlerta) {
      s.alertouFadiga = true;
      setTimeout(() => falar("A velocidade caiu bastante. Considere encerrar a série."), 900);
    }
  }
  definirStatus("Em pé", "ok");
}

// ---------- desenho ----------
function desenhar(pontos, W, H, falhas) {
  ctx.clearRect(0, 0, W, H);
  if (!pontos) return;
  const cor = falhas.length ? "#ff5a5f" : "#2ecc71";
  ctx.lineWidth = Math.max(3, W / 160);
  ctx.strokeStyle = cor;
  ctx.lineCap = "round";
  for (const [a, b] of LIGACOES) {
    const pa = pontos[a], pb = pontos[b];
    if ((pa.visibility ?? 1) < 0.4 || (pb.visibility ?? 1) < 0.4) continue;
    ctx.beginPath(); ctx.moveTo(pa.x * W, pa.y * H); ctx.lineTo(pb.x * W, pb.y * H); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const i of [11, 12, 23, 24, 25, 26, 27, 28]) {
    const p = pontos[i];
    if ((p.visibility ?? 1) < 0.4) continue;
    ctx.beginPath(); ctx.arc(p.x * W, p.y * H, Math.max(4, W / 110), 0, Math.PI * 2); ctx.fill();
  }
}

function laco() {
  if (!rodando) return;
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== ultimoTempoVideo) {
    ultimoTempoVideo = v.currentTime;
    const W = v.videoWidth, H = v.videoHeight;
    if (els.canvas.width !== W) { els.canvas.width = W; els.canvas.height = H; }
    const agora = performance.now();
    const res = detector.detectForVideo(v, agora);
    const pontos = res.landmarks && res.landmarks[0];
    if (pontos) {
      const r = analisar(pontos, W, H, agora);
      desenhar(pontos, W, H, r.falhas);
    } else {
      desenhar(null, W, H, []);
      definirStatus("Ninguém na imagem", "atencao");
      els.vista.textContent = "Entre no enquadramento";
    }
  }
  requestAnimationFrame(laco);
}

// ---------- fluxo ----------
async function iniciarSerie() {
  falar(" ", true); // libera a voz no iPhone (precisa de um toque)
  serie = novaSerie();
  els.reps.textContent = "0"; els.angulo.textContent = "–"; els.subida.textContent = "–";
  els.alerta.hidden = true;
  mostrarTela("treino");
  definirStatus("Abrindo câmera…", "atencao");
  els.vista.textContent = "";

  try {
    const modo = els.camera.value;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: modo, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    els.video.srcObject = stream;
    await els.video.play();
    const espelhar = modo === "user";
    els.video.classList.toggle("espelhado", espelhar);
    els.canvas.classList.toggle("espelhado", espelhar);
  } catch (e) {
    console.error(e);
    alert("Não consegui abrir a câmera. Verifique se o app tem permissão para usar a câmera (Ajustes do iPhone).");
    pararCamera(); mostrarTela("inicio"); return;
  }

  try {
    definirStatus("Carregando inteligência…", "atencao");
    els.vista.textContent = "Na primeira vez pode levar alguns segundos";
    await carregarDetector();
  } catch (e) {
    console.error(e);
    alert("Não consegui carregar a análise de movimento. Verifique a internet e tente de novo.");
    pararCamera(); mostrarTela("inicio"); return;
  }

  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* sem suporte */ }
  rodando = true;
  ultimoTempoVideo = -1;
  requestAnimationFrame(laco);
}

function pararCamera() {
  rodando = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  els.video.srcObject = null;
  try { wakeLock?.release(); } catch { /* nada */ }
  wakeLock = null;
}

function sugerir(s) {
  const reps = s.reps;
  if (!reps.length) return null;
  const boas = reps.filter((r) => !r.falhas.length).length;
  const pctBoas = boas / reps.length;
  const melhor = Math.max(...reps.slice(0, 3).map((r) => r.velocidade));
  const ultima = reps[reps.length - 1].velocidade;
  const perda = reps.length >= 3 && melhor > 0 ? Math.max(0, 1 - ultima / melhor) : null;
  const p = s.peso;

  if (pctBoas < 0.6) return { tipo: "baixar", peso: arred25(p * 0.9), perda, boas,
    texto: "Mais da metade das repetições teve erro de execução. Reduza um pouco o peso e foque na técnica antes de progredir." };
  if (perda == null) return { tipo: "manter", peso: p, perda, boas,
    texto: "Poucas repetições para medir o cansaço. Faça ao menos 3 repetições para o app sugerir a progressão." };
  if (perda < 0.15) {
    const inc = p >= 60 && perda < 0.10 ? 5 : 2.5;
    return { tipo: "subir", peso: arred25(p + inc), perda, boas,
      texto: `A velocidade quase não caiu (${Math.round(perda * 100)}%) e a técnica se manteve: ainda havia bastante reserva. Pode subir ${fmt(inc)} kg na próxima série.` };
  }
  if (perda <= 0.35) return { tipo: "manter", peso: p, perda, boas,
    texto: `Perda de velocidade de ${Math.round(perda * 100)}%: esforço na faixa ideal para ganho de massa muscular. Mantenha o peso.` };
  return { tipo: "baixar", peso: arred25(p * 0.95), perda, boas,
    texto: `A velocidade caiu ${Math.round(perda * 100)}%: sinal de muito cansaço. Reduza um pouco o peso ou descanse mais entre as séries.` };
}

function encerrarSerie() {
  pararCamera();
  speechSynthesis?.cancel();
  const s = serie;
  const sug = sugerir(s);
  let html = "";
  if (!s.reps.length) {
    html = `<p class="vazio">Nenhuma repetição registrada. Confira se o corpo inteiro aparece na câmera e se você esperou o aviso "Pronto".</p>`;
  } else {
    html += `<div class="res-grade">
      <div class="res-num"><b>${s.reps.length}</b><span>repetições</span></div>
      <div class="res-num"><b>${sug.boas}</b><span>com boa técnica</span></div>
      <div class="res-num"><b>${fmt(s.peso)} kg</b><span>peso usado</span></div>
      <div class="res-num"><b>${sug.perda == null ? "–" : Math.round(sug.perda * 100) + "%"}</b><span>queda de velocidade</span></div>
    </div>
    <div class="sugestao ${sug.tipo}">
      <b>Próxima série: ${fmt(sug.peso)} kg</b><br>${sug.texto}
    </div>
    <table class="tabela-reps"><tr><th>#</th><th>Subida</th><th>Joelho</th><th>Execução</th></tr>
    ${s.reps.map((r) => `<tr><td>${r.n}</td><td>${fmt(r.tempoSubida)} s</td><td>${Math.round(r.anguloMin)}°</td>
      <td class="${r.falhas.length ? "erro-txt" : "ok-txt"}">${r.falhas.length ? r.falhas.join(", ") : "Boa"}</td></tr>`).join("")}
    </table>`;
    salvarHistorico({ data: s.inicio, exercicio: "Agachamento", peso: s.peso, reps: s.reps.length, boas: sug.boas,
      perda: sug.perda, proximo: sug.peso, tipo: sug.tipo });
    els.peso.value = sug.peso;
  }
  els.resumo.innerHTML = html;
  mostrarTela("resumo");
  renderHistorico();
}

// ---------- histórico ----------
function lerHistorico() {
  try { return JSON.parse(localStorage.getItem(CHAVE_HIST)) || []; } catch { return []; }
}
function salvarHistorico(item) {
  try {
    const h = lerHistorico(); h.unshift(item);
    localStorage.setItem(CHAVE_HIST, JSON.stringify(h.slice(0, 200)));
  } catch { /* armazenamento indisponível */ }
}
function renderHistorico() {
  const h = lerHistorico();
  if (!h.length) { els.historico.innerHTML = `<p class="vazio">Nenhuma série ainda.</p>`; return; }
  const seta = { subir: "⬆️", manter: "➡️", baixar: "⬇️" };
  els.historico.innerHTML = h.slice(0, 15).map((i) => {
    const d = new Date(i.data);
    return `<div class="item-hist">
      <div class="l1"><span>${i.exercicio} · ${fmt(i.peso)} kg × ${i.reps}</span><span>${d.toLocaleDateString("pt-BR")} ${d.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })}</span></div>
      <div class="l2">${i.boas} boas · queda de velocidade ${i.perda == null ? "–" : Math.round(i.perda * 100) + "%"} · próxima ${seta[i.tipo] || ""} ${fmt(i.proximo)} kg</div>
    </div>`;
  }).join("");
}

// ---------- eventos ----------
$("btn-iniciar").addEventListener("click", iniciarSerie);
$("btn-encerrar").addEventListener("click", encerrarSerie);
$("btn-nova").addEventListener("click", iniciarSerie);
$("btn-voltar").addEventListener("click", () => mostrarTela("inicio"));
$("btn-limpar").addEventListener("click", () => {
  if (confirm("Apagar todo o histórico deste aparelho?")) { try { localStorage.removeItem(CHAVE_HIST); } catch {} renderHistorico(); }
});
document.querySelectorAll(".btn-peso").forEach((b) => b.addEventListener("click", () => {
  const v = parseFloat(String(els.peso.value).replace(",", ".")) || 0;
  els.peso.value = Math.max(0, v + parseFloat(b.dataset.delta));
}));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && rodando) navigator.wakeLock?.request("screen").then((w) => (wakeLock = w)).catch(() => {});
});

// Peso inicial = sugestão da última série
const ult = lerHistorico()[0];
if (ult) els.peso.value = ult.proximo;
renderHistorico();

// Aviso para instalar no iPhone quando aberto pelo Safari
const ehIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
const instalado = window.navigator.standalone === true || matchMedia("(display-mode: standalone)").matches;
if (ehIOS && !instalado) $("aviso-instalar").hidden = false;

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

// Já começa a baixar a inteligência em segundo plano
carregarDetector().catch(() => {});
