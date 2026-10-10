// Guarda o app e a inteligência de movimento no aparelho, para funcionar com internet fraca na academia.
const CACHE_APP = "formacerta-app-v1";
const CACHE_EXTERNO = "formacerta-externo-v1";
const ARQUIVOS_APP = ["./", "index.html", "estilo.css?v=1", "app.js?v=1", "manifest.webmanifest", "icone-180.png", "icone-192.png", "icone-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE_APP).then((c) => c.addAll(ARQUIVOS_APP)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((chaves) => Promise.all(
    chaves.filter((k) => k !== CACHE_APP && k !== CACHE_EXTERNO).map((k) => caches.delete(k))
  )).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;

  // Bibliotecas e modelo (não mudam, versão fixa): usa o que está guardado
  if (url.hostname === "cdn.jsdelivr.net" || url.hostname === "storage.googleapis.com") {
    e.respondWith(caches.open(CACHE_EXTERNO).then(async (c) => {
      const guardado = await c.match(e.request);
      if (guardado) return guardado;
      const resp = await fetch(e.request);
      if (resp.ok) c.put(e.request, resp.clone());
      return resp;
    }));
    return;
  }

  // Arquivos do próprio app: tenta a internet primeiro (para pegar atualizações), senão usa o guardado
  if (url.origin === self.location.origin) {
    e.respondWith(fetch(e.request).then((resp) => {
      if (resp.ok) caches.open(CACHE_APP).then((c) => c.put(e.request, resp.clone()));
      return resp;
    }).catch(() => caches.match(e.request).then((r) => r || caches.match("index.html"))));
  }
});
