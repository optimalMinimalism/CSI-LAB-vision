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
  await rpc("Browser.setDownloadBehavior", {
    behavior: "allow",
    downloadPath: temp + "/downloads",
  });
  await rpc("Page.navigate", { url: "http://127.0.0.1:" + port });
  await wait(
    () =>
      evaluate(
        "document.readyState==='complete'&&!!document.getElementById('token')",
      ),
    "pagina",
  );
  await evaluate(
    `document.getElementById('token').value=${JSON.stringify(token)};document.getElementById('login').requestSubmit()`,
  );
  await wait(
    () =>
      evaluate(
        "!document.getElementById('app').classList.contains('hidden')&&!busy",
      ),
    "accesso",
  );
  await wait(() => evaluate("cameraReady&&!busy&&document.getElementById('preview').naturalWidth>0"), "anteprima senza registrazione");
  await wait(() => evaluate("document.getElementById('presenceState').textContent.includes('sì')"), "presenza live");
  assert.equal(await evaluate("document.getElementById('postureState').textContent.includes('in piedi')"),true);
  assert.equal(await evaluate("document.getElementById('skeletonState').textContent.includes('1')"),true);
  assert.equal(await evaluate("/Acquisito: .*\\.\\d{9} UTC/.test(document.getElementById('previewTime').textContent)"),true);
  for (const [button, mode, pressed] of [['modePosture','presence,posture',2],['modeSkeleton','all',4],['modePresence','posture,skeleton',2],['modeAll','all',4],['modePosture','presence,skeleton',2]]) {
    await click(button);
    await wait(() => evaluate("document.getElementById('preview').dataset.mode==="+JSON.stringify(mode)), 'modalità '+mode);
    assert.equal(await evaluate("document.getElementById("+JSON.stringify(button)+").getAttribute('aria-pressed')"),['presence,skeleton','posture,skeleton'].includes(mode)?'false':'true');
    assert.equal(await evaluate("document.querySelectorAll('.preview-modes button[aria-pressed=\"true\"]').length"),pressed);
  }
  assert.equal(await evaluate("active"), null);
  assert.equal(await evaluate("api('recordings').then(records=>records.length)"), 0);
  assert.equal(await evaluate("!!document.getElementById('infer')||!!document.getElementById('labels')||!!document.getElementById('manifest')"), false);
  await evaluate("document.getElementById('title').value='Prova video';document.getElementById('startForm').requestSubmit()");
  await wait(() => evaluate("active&&!busy"), "avvio video");
  const identifier = await evaluate("active.id");
  await wait(() => evaluate("api('recordings/" + identifier + "').then(item=>item.frames>=3)"), "frame registrati");
  await click("stop");
  await wait(() => evaluate("!busy&&!active"), "arresto video");
  assert.equal(await evaluate("cameraReady"), true);
  assert.equal(await evaluate("api('recordings/" + identifier + "').then(item=>'csi_samples' in item)"), false);
  await evaluate("document.querySelector('#recordings button').click()");
  const download = await wait(async () => {
    const files = await fs.readdir(temp + "/downloads");
    return files.find(file => file.endsWith('.mp4'));
  }, "download MP4");
  const video = await fs.readFile(temp + '/downloads/' + download);
  assert(video.length > 1000);
  assert(video.subarray(0,32).includes(Buffer.from('ftyp')));
  const screenshot = await rpc("Page.captureScreenshot", {format:"png"});
  await fs.writeFile(temp + '/preview.png', Buffer.from(screenshot.data,'base64'));
  await rpc("Emulation.setDeviceMetricsOverride", {width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate("document.documentElement.scrollWidth<=window.innerWidth+1"),true);
  await click('logout');
  assert.equal(await evaluate("document.getElementById('app').classList.contains('hidden')"),true);
  assert.deepEqual(runtimeErrors, []);
  console.log(JSON.stringify({result:'ok',checks:['accesso','anteprima senza salvataggio','registrazione solo video','camera-only recording metadata','download MP4','anteprima dopo stop','presenza/postura/scheletro live','timestamp acquisizione esatto','overlay combinabili e pulsante All','mobile','logout'],screenshot:temp+'/preview.png'}));
} finally {
  if (ws) ws.close();
  if (browser) browser.kill("SIGTERM");
  if (python) python.kill("SIGTERM");
  // Test artifacts are deliberately retained in /tmp, never in source control.
}
