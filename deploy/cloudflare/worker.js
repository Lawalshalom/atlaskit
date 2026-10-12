// Public front door for a RunPod Serverless AtlasKit endpoint (Cloudflare Workers).
//
// RunPod load-balancing endpoints require RunPod's own API key on every request, and that key must
// never reach a browser. This Worker keeps it as a secret (RUNPOD_KEY), moves the caller's AtlasKit
// key into X-API-Key (the gateway checks X-API-Key first) and streams the response back. Bindings:
//   BACKEND     https://<endpoint-id>.api.runpod.ai
//   PLAYGROUND  where "/" redirects, e.g. https://lawalshalom.github.io/atlaskit/
//   RUNPOD_KEY  secret: a RunPod API key restricted to this endpoint
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "authorization, content-type, x-api-key",
  "access-control-max-age": "600",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    // Answer CORS preflights here, so they never wait for a sleeping GPU.
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
    if (url.pathname === "/" && env.PLAYGROUND) return Response.redirect(env.PLAYGROUND, 302);

    const headers = new Headers(request.headers);
    const auth = headers.get("authorization") || "";
    if (!headers.has("x-api-key") && /^bearer\s+/i.test(auth)) headers.set("x-api-key", auth.replace(/^bearer\s+/i, "").trim());
    headers.set("authorization", `Bearer ${env.RUNPOD_KEY}`);
    for (const h of ["host", "cf-connecting-ip", "cf-ipcountry", "cf-ray", "cf-visitor", "x-forwarded-proto", "x-real-ip"]) headers.delete(h);

    const upstream = await fetch(new URL(url.pathname + url.search, env.BACKEND), {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "manual",
    });
    const response = new Response(upstream.body, upstream);
    for (const [k, v] of Object.entries(CORS)) response.headers.set(k, v);
    return response;
  },
};
