// Browser smoke test, Node 20 --experimental-websocket or Node 22+, no npm dependency.
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
const root = path.resolve(
  import.meta.dirname ?? path.dirname(new URL(import.meta.url).pathname),
  "..",
);
const temp = await fs.mkdtemp("/tmp/csi-video-browser-"),
  token = randomBytes(32).toString("hex");
let python,
  browser,
  ws,
  serial = 0,
  logs = "",
  runtimeErrors = [];
const pending = new Map();
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function wait(fn, label) {
  for (let i = 0; i < 200; i++) {
    try {
      const value = await fn();
      if (value) return value;
    } catch {}
    await delay(100);
  }
  throw Error("Timeout: " + label + " " + logs.slice(-1500));
}
function rpc(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await rpc("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails)
    throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function click(id) {
  await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
}
try {
  python = spawn(
    process.env.CSI_VIDEO_TEST_PYTHON ?? "python3",
    [root + "/tests/browserFixture.py"],
    {
      cwd: root,
      env: {
        ...process.env,
        CSI_VIDEO_TEST_DATA: temp + "/data",
        CSI_VIDEO_TEST_DISTANCE: "true",
        CSI_VIDEO_TOKEN: token,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let stdout = "";
  python.stdout.on("data", (data) => (stdout += data));
  python.stderr.on("data", (data) => (logs += data));
  const port = await wait(() => {
    const line = stdout.split("\n").find((line) => line.startsWith("{"));
    return line && JSON.parse(line).port;
  }, "receiver");
  const binary =
    process.env.CSI_VIDEO_TEST_BROWSER ??
    "/snap/chromium/current/usr/lib/chromium-browser/chrome";
  browser = spawn(
    binary,
    [
      "--headless",
      "--no-sandbox",
      "--disable-gpu",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--disable-background-networking",
      "--remote-debugging-address=127.0.0.1",
      "--remote-debugging-port=0",
      "--user-data-dir=" + temp + "/profile",
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  browser.stderr.on("data", (data) => (logs += data));
  const debugPort = await wait(
    async () =>
      Number(
        (await fs.readFile(temp + "/profile/DevToolsActivePort", "utf8")).split(
          "\n",
        )[0],
      ),
    "Chromium",
  );
  const target = (
    await (await fetch("http://127.0.0.1:" + debugPort + "/json/list")).json()
  ).find((item) => item.type === "page");
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.method === "Runtime.exceptionThrown")
      runtimeErrors.push(message.params);
    const call = pending.get(message.id);
    if (call) {
      pending.delete(message.id);
      message.error
        ? call.reject(Error(JSON.stringify(message.error)))
        : call.resolve(message.result);
    }
  };
  await rpc("Page.enable");
  await rpc("Runtime.enable");
  await rpc("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1100,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await rpc("Page.navigate", { url: "http://127.0.0.1:" + port + "/distance" });
  await wait(() => evaluate("document.readyState==='complete'&&!!document.getElementById('token')"), "distance page");
  await evaluate(`document.getElementById('token').value=${JSON.stringify(token)};document.getElementById('login').requestSubmit()`);
  await wait(() => evaluate("!document.getElementById('layout').hidden"), "login");
  assert.equal(await evaluate("running"), false);
  assert.equal(await evaluate("document.title.includes('[experimental]')"), true);
  await click('start');
  await wait(() => evaluate("metadata&&metadata.floor_distances.state==='uncalibrated'&&!polling"), "uncalibrated model");
  assert.equal(await evaluate("document.querySelectorAll('#measurements tr').length"), 0);
  await click('mark');
  for (const point of [[20,200],[300,200],[300,80],[20,80]]) {
    await evaluate(`(()=>{const canvas=document.getElementById('view'),r=canvas.getBoundingClientRect();canvas.dispatchEvent(new MouseEvent('click',{clientX:r.left+${point[0]}*r.width/canvas.width,clientY:r.top+${point[1]}*r.height/canvas.height,bubbles:true}))})()`);
  }
  assert.equal(await evaluate("points.length"), 4);
  await evaluate("document.getElementById('widthM').value='4';document.getElementById('depthM').value='3';document.getElementById('floorConfirmed').checked=true");
  await click('save');
  await wait(() => evaluate("metadata&&metadata.floor_distances.state==='ready'&&document.querySelectorAll('#measurements tr').length>0"), "calibrated distances");
  assert.equal(await evaluate("Math.abs(metadata.floor_distances.distances[0].distance_m-2)<.01"), true);
  const screenshot = await rpc('Page.captureScreenshot',{format:'png'});
  await fs.writeFile(temp+'/distance.png',Buffer.from(screenshot.data,'base64'));
  await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate("document.documentElement.scrollWidth<=window.innerWidth+1"),true);
  // The main preview retains its own annotated feed and preempts the experiment.
  await evaluate("fetch('/api/camera/preview',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({enabled:true})})");
  await wait(() => evaluate("document.getElementById('status').textContent.includes('In pausa')"), 'main preview priority');
  const primary = await evaluate("fetch('/api/camera/preview?mode=all',{headers:{Authorization:'Bearer '+token}}).then(r=>JSON.parse(r.headers.get('X-Frame-Metadata')))");
  assert.equal(primary.preview_mode,'all');
  assert.equal(primary.analysis.presence.state,'PRESENT');
  assert.equal('floor_distances' in primary,false);
  await evaluate("fetch('/api/camera/preview',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({enabled:false})})");
  await wait(() => evaluate("metadata&&metadata.floor_distances.state==='ready'"), 'resume after main preview');
  const recording = await evaluate("fetch('/api/recordings',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({title:'Primary recording during experiment'})}).then(r=>r.json())");
  await wait(() => evaluate("document.getElementById('status').textContent.includes('In pausa')"), 'recording priority');
  await wait(() => evaluate(`fetch('/api/recordings/${recording.id}',{headers:{Authorization:'Bearer '+token}}).then(r=>r.json()).then(r=>r.frames>=2)`), 'recording continues');
  await evaluate(`fetch('/api/recordings/${recording.id}/stop',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:'{}'})`);
  await wait(() => evaluate("metadata&&metadata.floor_distances.state==='ready'"), 'resume after recording');
  await click('reset');
  await wait(() => evaluate("metadata&&metadata.floor_distances.state==='uncalibrated'&&document.querySelectorAll('#measurements tr').length===0"), 'clear calibration');
  await click('stop');
  await wait(() => evaluate("!running&&document.getElementById('stop').disabled"), "stop");
  assert.equal(await evaluate("document.querySelectorAll('#measurements tr').length"),0);
  await click('logout');
  await wait(() => evaluate("document.getElementById('layout').hidden"), "logout");
  assert.deepEqual(runtimeErrors,[]);
  console.log(JSON.stringify({result:'ok',checks:['login','on-demand start','no uncalibrated metres','four-point calibration','known floor distance','mobile','main annotated preview priority','recording continues while experiment paused','automatic resume','clear calibration','stop','logout'],screenshot:temp+'/distance.png'}));
} finally {
  if (ws) ws.close();
  if (browser) browser.kill("SIGTERM");
  if (python) python.kill("SIGTERM");
  // Test artifacts are deliberately retained in /tmp, never in source control.
}
