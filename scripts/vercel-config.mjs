export function deploymentConfig(value) {
  if (!value) throw new Error('Set BACKEND_ORIGIN to the HTTPS origin of the running Python backend before deploying.');
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || url.pathname !== '/' || url.search || url.hash || url.hostname === 'localhost' || url.hostname === '127.0.0.1') {
    throw new Error('BACKEND_ORIGIN must be an HTTPS origin without credentials, path, query or fragment.');
  }
  return {version: 3, routes: [
    {src: '/(api|assets-store)(/.*)?', dest: `${url.origin}/$1$2`, headers: {'Cache-Control': 'private, no-store'}},
    {handle: 'filesystem'},
    {src: '/.*', dest: '/index.html'}
  ]};
}
