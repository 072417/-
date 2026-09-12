import test from 'node:test';
import assert from 'node:assert/strict';
import {deploymentConfig} from '../scripts/vercel-config.mjs';
test('reject missing or unsafe backend origins', () => {
 for (const value of [undefined, '', 'http://backend.example', 'https://user:password@backend.example', 'https://backend.example/path', 'https://backend.example/?key=secret']) assert.throws(() => deploymentConfig(value));
});
test('proxy private routes before the static fallback', () => {
 const config = deploymentConfig('https://backend.example');
 const route = config.routes[0];
 const match = new RegExp(`^${route.src}$`);
 for (const path of ['/api/runs', '/api/runs/id/export/json', '/assets-store/id/image.png']) {
   assert.ok(match.test(path));
   assert.equal(path.replace(match, route.dest), `https://backend.example${path}`);
 }
 assert.equal(match.test('/assets/app.js'), false);
 assert.equal(route.headers['Cache-Control'], 'private, no-store');
 assert.deepEqual(config.routes[1], {handle: 'filesystem'});
});
