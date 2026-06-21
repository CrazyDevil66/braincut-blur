HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainCut Blur</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#000;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;height:100vh;overflow:hidden;-webkit-font-smoothing:antialiased}
.app{display:flex;height:100vh}
.sidebar{width:268px;flex-shrink:0;background:#0d0d0d;border-right:1px solid rgba(255,255,255,.07);display:flex;flex-direction:column;overflow-y:auto;padding-bottom:24px}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:2px}
.sb-logo{display:flex;align-items:center;gap:11px;padding:20px 18px 16px;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0}
.logo-box{width:36px;height:36px;flex-shrink:0;background:linear-gradient(135deg,#0a84ff,#5e5ce6);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1.2rem}
.logo-title{font-size:.95rem;font-weight:700;letter-spacing:-.02em}
.logo-sub{font-size:.63rem;color:rgba(255,255,255,.3);margin-top:1px}
.sb-section{padding:16px 16px 0}
.sb-label{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:rgba(255,255,255,.25);margin-bottom:9px}
.hw-chips{display:flex;flex-wrap:wrap;gap:5px}
.hw-chip{display:flex;align-items:center;gap:5px;padding:4px 9px;border-radius:20px;font-size:.67rem;font-weight:600;background:#1a1a1a;border:1px solid rgba(255,255,255,.09);color:rgba(255,255,255,.3);transition:all .35s}
.hw-chip.on{border-color:rgba(48,209,88,.38);color:#30d158;background:rgba(48,209,88,.07)}
.hw-dot{width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}
.model-active-row{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:10px;background:#1a1a1a;border:1px solid rgba(255,255,255,.06);margin-bottom:6px}
.model-active-icon{font-size:.95rem;flex-shrink:0}
.model-active-name{font-size:.76rem;font-weight:600;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.model-active-badge{font-size:.57rem;font-weight:700;padding:2px 5px;border-radius:4px;background:rgba(10,132,255,.18);color:#0a84ff;flex-shrink:0}
.manage-btn{width:100%;padding:7px;border-radius:8px;border:1px solid rgba(255,255,255,.09);background:transparent;color:rgba(255,255,255,.4);font-size:.71rem;font-weight:500;cursor:pointer;transition:all .2s;font-family:inherit;margin-top:4px}
.manage-btn:hover{border-color:rgba(255,255,255,.22);color:#fff}
.slider-row{margin-bottom:13px}
.slider-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}
.slider-label{font-size:.7rem;color:rgba(255,255,255,.55)}
.slider-val{font-size:.7rem;font-weight:700;color:#0a84ff}
input[type=range]{width:100%;appearance:none;height:3px;border-radius:2px;background:rgba(255,255,255,.12);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{appearance:none;width:13px;height:13px;border-radius:50%;background:#0a84ff;cursor:pointer;transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.25)}
.hf-row{display:flex;gap:6px}
.hf-input{flex:1;background:#1a1a1a;border:1px solid rgba(255,255,255,.1);border-radius:7px;padding:7px 10px;color:#fff;font-size:.7rem;font-family:inherit;outline:none;min-width:0}
.hf-input:focus{border-color:#0a84ff}
.hf-save{padding:7px 10px;border-radius:7px;border:none;background:rgba(10,132,255,.18);color:#0a84ff;font-size:.7rem;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;transition:background .15s}
.hf-save:hover{background:rgba(10,132,255,.32)}
.main{flex:1;display:flex;flex-direction:column;overflow-y:auto;padding:20px 22px;gap:11px;min-width:0}
.main::-webkit-scrollbar{width:4px}
.main::-webkit-scrollbar-thumb{background:#2c2c2e;border-radius:2px}
.main-top{display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.status-pill{display:flex;align-items:center;gap:6px;padding:5px 13px;background:#1c1c1e;border-radius:20px;font-size:.76rem;font-weight:500;border:1px solid rgba(255,255,255,.08)}
.status-dot{width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,.22);flex-shrink:0}
.status-dot.active{background:#30d158}
.status-dot.error{background:#ff453a}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.pulse{animation:pulse 1.6s ease-in-out infinite}
.cancel-btn{background:transparent;border:1px solid rgba(255,69,58,.38);color:#ff453a;padding:5px 14px;border-radius:8px;font-size:.74rem;font-weight:600;cursor:pointer;transition:all .2s}
.cancel-btn:hover:not(:disabled){background:rgba(255,69,58,.1);border-color:#ff453a}
.cancel-btn:disabled{opacity:.4;cursor:not-allowed}
.card{background:#1c1c1e;border-radius:16px;padding:16px 18px;border:1px solid rgba(255,255,255,.06);flex-shrink:0}
.card-title{font-size:.6rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:rgba(255,255,255,.3);margin-bottom:11px}
.err-banner{background:rgba(255,69,58,.1);border:1px solid rgba(255,69,58,.28);border-radius:12px;padding:11px 15px;color:#ff453a;font-size:.8rem;gap:8px;align-items:flex-start;flex-shrink:0}
.job-name{font-size:.93rem;font-weight:600;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.job-sub{font-size:.71rem;color:rgba(255,255,255,.33);margin-bottom:13px}
.progress-track{height:4px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;margin-bottom:6px}
.progress-fill{height:100%;background:#0a84ff;border-radius:2px;transition:width .5s ease;position:relative;overflow:hidden}
.progress-fill::after{content:'';position:absolute;top:0;left:-40px;width:40px;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.38),transparent);animation:shimmer 1.6s linear infinite}
@keyframes shimmer{to{left:110%}}
.progress-meta{display:flex;justify-content:space-between;font-size:.69rem;color:rgba(255,255,255,.28);margin-bottom:14px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:13px}
.metric{background:#252525;border-radius:10px;padding:10px 12px}
.metric-lbl{font-size:.59rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:rgba(255,255,255,.28);margin-bottom:4px}
.metric-val{font-size:1.05rem;font-weight:700}
.metric-val.blue{color:#0a84ff}
.metric-val.green{color:#30d158}
.spark-wrap{background:#111;border-radius:8px;padding:5px 8px 4px;margin-bottom:12px;position:relative;height:50px}
.spark-wrap svg{width:100%;height:36px;display:block;overflow:visible}
.spark-meta{display:flex;justify-content:space-between;font-size:.57rem;color:rgba(255,255,255,.18);margin-top:1px}
.det-row{display:flex;gap:8px}
.det-card{flex:1;background:#252525;border-radius:10px;padding:9px 12px;display:flex;align-items:center;gap:9px}
.det-icon{font-size:1.05rem}
.det-num{font-size:1rem;font-weight:700}
.det-lbl{font-size:.62rem;color:rgba(255,255,255,.32);margin-top:1px}
.render-row{display:flex;align-items:center;gap:12px}
.spinner{width:18px;height:18px;flex-shrink:0;border:2px solid rgba(255,255,255,.1);border-top-color:#0a84ff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.log-box{background:#000;border-radius:10px;height:230px;overflow-y:auto;font-family:'SF Mono','Menlo','Monaco','Consolas',monospace;font-size:.69rem;color:rgba(255,255,255,.26);padding:10px 12px}
.log-box::-webkit-scrollbar{width:3px}
.log-box::-webkit-scrollbar-thumb{background:#2c2c2e;border-radius:2px}
.le{padding:1.5px 0;line-height:1.65}
.le:last-child{color:rgba(255,255,255,.75)}
.le .ts{color:rgba(255,255,255,.17)}
.le.w{color:#ffd60a}
.le.e{color:#ff453a}
.log-foot{display:flex;justify-content:space-between;margin-top:7px;font-size:.66rem;color:rgba(255,255,255,.2)}
.as-btn{cursor:pointer;transition:color .2s}
.as-btn.on{color:#0a84ff}
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:50;opacity:0;pointer-events:none;transition:opacity .3s}
.scrim.open{opacity:1;pointer-events:auto}
.drawer{position:fixed;top:0;right:0;width:420px;max-width:100vw;height:100%;background:#1c1c1e;z-index:51;transform:translateX(100%);transition:transform .35s cubic-bezier(.4,0,.2,1);overflow-y:auto;display:flex;flex-direction:column;pointer-events:none}
.drawer.open{transform:translateX(0);pointer-events:auto}
.drawer-header{display:flex;align-items:center;justify-content:space-between;padding:20px 20px 16px;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0;position:sticky;top:0;background:#1c1c1e;z-index:1}
.drawer-title{font-size:1rem;font-weight:600}
.drawer-close{background:#2c2c2e;border:none;color:rgba(255,255,255,.6);width:28px;height:28px;border-radius:50%;font-size:.75rem;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s}
.drawer-close:hover{background:#3a3a3c;color:#fff}
.drawer-body{padding:16px 20px;flex:1}
.catalog-bar{display:flex;align-items:center;justify-content:space-between;background:#2c2c2e;border-radius:10px;padding:10px 14px;margin-bottom:16px}
.catalog-info{font-size:.75rem;color:rgba(255,255,255,.45);flex:1;min-width:0}
.refresh-btn{background:transparent;border:1px solid rgba(255,255,255,.15);color:rgba(255,255,255,.65);padding:4px 12px;border-radius:7px;font-size:.72rem;font-weight:500;cursor:pointer;transition:all .2s;flex-shrink:0;margin-left:10px;font-family:inherit}
.refresh-btn:hover:not(:disabled){border-color:#0a84ff;color:#0a84ff}
.refresh-btn:disabled{opacity:.4;cursor:not-allowed}
.seg-ctrl{display:flex;background:#2c2c2e;border-radius:9px;padding:2px;margin-bottom:14px}
.seg-btn{flex:1;padding:7px 12px;border-radius:7px;border:none;background:transparent;color:rgba(255,255,255,.45);font-size:.8rem;font-weight:500;cursor:pointer;transition:all .2s;text-align:center;font-family:inherit}
.seg-btn.active{background:#3a3a3c;color:#fff;font-weight:600}
.model-list{display:flex;flex-direction:column;gap:8px}
.model-row{display:flex;align-items:center;gap:12px;background:#2c2c2e;border-radius:12px;padding:12px 14px;border:1px solid transparent;transition:border-color .2s}
.model-row.active{border-color:#0a84ff}
.model-icon{width:36px;height:36px;border-radius:8px;background:#3a3a3c;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0}
.model-icon.fi{background:rgba(10,132,255,.15)}
.model-icon.pi{background:rgba(48,209,88,.15)}
.model-info{flex:1;min-width:0}
.model-name{font-size:.85rem;font-weight:600}
.model-desc{font-size:.72rem;color:rgba(255,255,255,.38);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.model-meta{font-size:.68rem;color:rgba(255,255,255,.22);margin-top:2px}
.active-badge{display:inline-block;background:rgba(10,132,255,.2);color:#0a84ff;font-size:.6rem;font-weight:700;letter-spacing:.05em;padding:2px 6px;border-radius:5px;text-transform:uppercase;margin-left:6px;vertical-align:middle}
.model-actions{display:flex;flex-direction:column;gap:5px;flex-shrink:0}
.m-btn{padding:5px 12px;border-radius:7px;border:none;font-size:.72rem;font-weight:600;cursor:pointer;transition:all .15s;white-space:nowrap;font-family:inherit}
.m-btn:disabled{opacity:.4;cursor:not-allowed}
.m-btn-install{background:rgba(10,132,255,.2);color:#0a84ff}
.m-btn-install:hover:not(:disabled){background:rgba(10,132,255,.35)}
.m-btn-activate{background:rgba(48,209,88,.15);color:#30d158}
.m-btn-activate:hover:not(:disabled){background:rgba(48,209,88,.28)}
.m-btn-delete{background:rgba(255,69,58,.1);color:#ff453a}
.m-btn-delete:hover:not(:disabled){background:rgba(255,69,58,.22)}
.dl-bar{height:3px;background:rgba(255,255,255,.1);border-radius:2px;margin-top:6px;overflow:hidden}
.dl-bar-fill{height:100%;background:#0a84ff;border-radius:2px;transition:width .3s}
</style>
</head>
<body>

<div class="scrim" id="scrim"></div>

<div class="drawer" id="drawer">
  <div class="drawer-header">
    <span class="drawer-title">Modelle verwalten</span>
    <button class="drawer-close" id="drawerClose">&#x2715;</button>
  </div>
  <div class="drawer-body">
    <div class="catalog-bar">
      <span class="catalog-info" id="catalogInfo">Lade Katalog&hellip;</span>
      <button class="refresh-btn" id="refreshCatalogBtn">&#x21bb; Aktualisieren</button>
    </div>
    <div class="seg-ctrl">
      <button class="seg-btn active" id="tabFace">Gesicht</button>
      <button class="seg-btn" id="tabPlate">Kennzeichen</button>
    </div>
    <div class="model-list" id="faceModelList"><div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Lade&hellip;</div></div>
    <div class="model-list" id="plateModelList" style="display:none"><div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Lade&hellip;</div></div>
  </div>
</div>

<div class="app">
  <aside class="sidebar">
    <div class="sb-logo">
      <div class="logo-box">&#9986;</div>
      <div><div class="logo-title">BrainCut Blur</div><div class="logo-sub">Face &amp; Plate Detection</div></div>
    </div>

    <div class="sb-section">
      <div class="sb-label">Hardware</div>
      <div class="hw-chips">
        <div class="hw-chip" id="chipNvdec"><span class="hw-dot"></span>NVDEC</div>
        <div class="hw-chip" id="chipNvenc"><span class="hw-dot"></span>NVENC</div>
        <div class="hw-chip" id="chipTrt"><span class="hw-dot"></span>TRT</div>
      </div>
    </div>

    <div class="sb-section" style="margin-top:16px">
      <div class="sb-label">Modelle</div>
      <div class="model-active-row" id="sbFaceModel">
        <span class="model-active-icon">&#128065;</span>
        <span class="model-active-name">&#8211;</span>
      </div>
      <div class="model-active-row" id="sbPlateModel">
        <span class="model-active-icon">&#128663;</span>
        <span class="model-active-name">&#8211;</span>
      </div>
      <button class="manage-btn" id="gearBtn">&#9881; Modelle verwalten</button>
    </div>

    <div class="sb-section" style="margin-top:16px">
      <div class="sb-label">Erkennung</div>
      <div class="slider-row">
        <div class="slider-top"><span class="slider-label">Frame-Skip</span><span class="slider-val" id="frameSkipVal">4</span></div>
        <input type="range" min="1" max="8" step="1" value="4" id="frameSkipSlider">
      </div>
      <div class="slider-row">
        <div class="slider-top"><span class="slider-label">Kennzeichen-Konfidenz</span><span class="slider-val" id="confVal">0.45</span></div>
        <input type="range" min="0.30" max="0.80" step="0.05" value="0.45" id="confSlider">
      </div>
    </div>

    <div class="sb-section" style="margin-top:16px">
      <div class="sb-label">HuggingFace Token</div>
      <div class="hf-row">
        <input type="password" class="hf-input" id="hfTokenInput" placeholder="hf_xxxx&hellip;">
        <button class="hf-save" id="hfTokenSaveBtn">Speichern</button>
      </div>
      <div id="hfTokenStatus" style="font-size:.66rem;color:rgba(255,255,255,.3);margin-top:5px"></div>
    </div>
  </aside>

  <main class="main">
    <div class="main-top">
      <div class="status-pill" id="pill">
        <div class="status-dot" id="sdot"></div>
        <span id="pillText">Laden&hellip;</span>
      </div>
      <button class="cancel-btn" id="cancelBtn" style="display:none">Abbrechen</button>
    </div>

    <div class="err-banner" id="errBanner" style="display:none">
      &#9888;&nbsp;<span id="errText"></span>
    </div>

    <div class="card" id="jobCard" style="display:none">
      <div class="job-name" id="jobName"></div>
      <div class="job-sub"  id="jobSub"></div>
      <div class="progress-track"><div class="progress-fill" id="pBar" style="width:0%"></div></div>
      <div class="progress-meta"><span id="pLeft"></span><span id="pRight"></span></div>
      <div class="metrics">
        <div class="metric"><div class="metric-lbl">Geschwindigkeit</div><div class="metric-val blue" id="sFps">&#8211;</div></div>
        <div class="metric"><div class="metric-lbl">Verstrichen</div><div class="metric-val" id="sElapsed">&#8211;</div></div>
        <div class="metric"><div class="metric-lbl">Verbleibend</div><div class="metric-val green" id="sEta">&#8211;</div></div>
      </div>
      <div class="spark-wrap">
        <svg id="sparkline" preserveAspectRatio="none"></svg>
        <div class="spark-meta"><span id="sparkMin"></span><span id="sparkPeak"></span></div>
      </div>
      <div class="det-row">
        <div class="det-card"><span class="det-icon">&#128065;</span><div><div class="det-num" id="faceCount">0</div><div class="det-lbl">Gesichter</div></div></div>
        <div class="det-card"><span class="det-icon">&#128663;</span><div><div class="det-num" id="plateCount">0</div><div class="det-lbl">Kennzeichen</div></div></div>
      </div>
    </div>

    <div class="card" id="renderCard" style="display:none">
      <div class="card-title">Encodiert</div>
      <div class="render-row">
        <div class="spinner"></div>
        <div>
          <div style="font-size:.9rem;font-weight:600" id="renderName"></div>
          <div style="font-size:.71rem;color:rgba(255,255,255,.35);margin-top:2px">NVENC H.264 &ndash; Encoding l&auml;uft&hellip;</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <span class="card-title" style="margin-bottom:0">Protokoll</span>
        <span id="logCount" style="font-size:.66rem;color:rgba(255,255,255,.2)">0 Eintr&auml;ge</span>
      </div>
      <div class="log-box" id="logBox"></div>
      <div class="log-foot">
        <span id="refreshTs"></span>
        <span class="as-btn on" id="asBtn">&#8595; Auto-Scroll</span>
      </div>
    </div>
  </main>
</div>

<script>
var lastF=0,lastFT=Date.now(),smoothFps=0,autoScroll=true;
var settingsData=null,activeTab='face',hfToken=localStorage.getItem('hf_token')||''
var fpsHistory=[],configData={detection_interval:4,plate_conf_thresh:0.45};

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmt(s){if(!s||s<=0)return '–';if(s<60)return s+' Sek';var m=Math.floor(s/60),r=s%60;return r?m+' min '+r+' s':m+' min';}
function calcFps(cf){
  var now=Date.now(),dt=(now-lastFT)/1000;
  if(dt>=1.8&&cf>lastF){var raw=(cf-lastF)/dt;smoothFps=smoothFps===0?raw:smoothFps*.55+raw*.45;lastF=cf;lastFT=now;}
  return smoothFps>0?smoothFps.toFixed(1)+' fps':'–';
}
function updateSparkline(fps){
  if(fps>0){fpsHistory.push(fps);if(fpsHistory.length>90)fpsHistory.shift();}
  var svg=document.getElementById('sparkline');
  if(!svg||fpsHistory.length<2)return;
  var W=svg.getBoundingClientRect().width||400,H=36;
  var max=Math.max.apply(null,fpsHistory),min=Math.min.apply(null,fpsHistory);
  if(max===0)return;
  var rng=max-min||1;
  var pts=fpsHistory.map(function(v,i){
    var x=(i/(fpsHistory.length-1))*W;
    var y=H-((v-min)/rng)*(H-6)-3;
    return x.toFixed(1)+','+y.toFixed(1);
  });
  var area='0,'+H+' '+pts.join(' ')+' '+W+','+H;
  svg.innerHTML='<polygon fill="rgba(10,132,255,.09)" points="'+area+'"/>'+
    '<polyline fill="none" stroke="#0a84ff" stroke-width="1.5" stroke-linejoin="round" points="'+pts.join(' ')+'"/>';
  var pk=document.getElementById('sparkPeak'),mn=document.getElementById('sparkMin');
  if(pk)pk.textContent='Peak '+max.toFixed(1)+' fps';
  if(mn)mn.textContent='Min '+min.toFixed(1)+' fps';
}
function setChip(id,on){var el=document.getElementById(id);if(el)el.className='hw-chip'+(on?' on':'')}
function logClass(l){if(/FEHLER|ERROR/i.test(l))return 'e';if(/Warnung|Warning|warn/i.test(l))return 'w';return '';}
function renderLog(l){return esc(l).replace(/(\\[\\d{2}:\\d{2}:\\d{2}\\])/,'<span class="ts">$1</span>');}
function toggleAs(){autoScroll=!autoScroll;document.getElementById('asBtn').className='as-btn'+(autoScroll?' on':'')}

async function doCancel(){
  var btn=document.getElementById('cancelBtn');btn.disabled=true;btn.textContent='Wird abgebrochen…';
  try{await fetch('/cancel',{method:'POST'});}catch(e){}
}
function saveHfToken(){
  hfToken=document.getElementById('hfTokenInput').value.trim();
  localStorage.setItem('hf_token',hfToken);
  var st=document.getElementById('hfTokenStatus');
  st.textContent=hfToken?'Token gespeichert.':'Token geleert.';
  st.style.color=hfToken?'#30d158':'rgba(255,255,255,.3)';
  setTimeout(function(){st.textContent='';},2500);
}
function openSettings(){document.getElementById('drawer').classList.add('open');document.getElementById('scrim').classList.add('open');loadSettings();}
function closeSettings(){document.getElementById('drawer').classList.remove('open');document.getElementById('scrim').classList.remove('open');}
function switchTab(tab){
  activeTab=tab;
  document.getElementById('tabFace').className='seg-btn'+(tab==='face'?' active':'');
  document.getElementById('tabPlate').className='seg-btn'+(tab==='plate'?' active':'');
  document.getElementById('faceModelList').style.display=tab==='face'?'':' none';
  document.getElementById('plateModelList').style.display=tab==='plate'?'':' none';
}
async function loadSettings(){
  try{var r=await fetch('/api/models');settingsData=await r.json();renderSettings();}
  catch(e){document.getElementById('faceModelList').innerHTML='<div style="color:#ff453a;font-size:.8rem">Fehler.</div>';}
}
function renderSettings(){
  var d=settingsData;if(!d)return;
  document.getElementById('catalogInfo').textContent=(d.catalog_source||'integriert')+' · '+d.catalog.length+' Modelle';
  renderModelList('face',d);renderModelList('plate',d);
  var faceId=d.config.face_model||'builtin-centerface';
  var plateId=d.config.plate_model||'';
  var faceM=d.catalog.find(function(m){return m.id===faceId;});
  var plateM=d.catalog.find(function(m){return m.id===plateId;});
  var sbf=document.getElementById('sbFaceModel');
  var sbp=document.getElementById('sbPlateModel');
  if(sbf)sbf.innerHTML='<span class="model-active-icon">&#128065;</span><span class="model-active-name">'+(faceM?esc(faceM.name):'–')+'</span><span class="model-active-badge">Aktiv</span>';
  if(sbp)sbp.innerHTML='<span class="model-active-icon">&#128663;</span><span class="model-active-name">'+(plateM?esc(plateM.name):(plateId?esc(plateId):'–'))+'</span>'+(plateM?'<span class="model-active-badge">Aktiv</span>':'');
}
function renderModelList(type,d){
  var listId=type==='face'?'faceModelList':'plateModelList';
  var models=d.catalog.filter(function(m){return m.type===type;});
  var activeId=type==='face'?d.config.face_model:d.config.plate_model;
  var iconCls=type==='face'?'fi':'pi',iconChar=type==='face'?'&#128065;':'&#128663;';
  if(!models.length){document.getElementById(listId).innerHTML='<div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Keine Modelle.</div>';return;}
  var html='';
  models.forEach(function(m){
    var isActive=m.id===activeId,isInstalled=!!(d.installed[m.id]),prog=d.install_progress[m.id],isDl=prog&&prog.status==='downloading';
    html+='<div class="model-row'+(isActive?' active':'')+'"><div class="model-icon '+iconCls+'">'+ iconChar +'</div>';
    html+='<div class="model-info"><div class="model-name">'+esc(m.name);
    if(isActive)html+='<span class="active-badge">Aktiv</span>';
    html+='</div><div class="model-desc">'+esc(m.description)+'</div>';
    html+='<div class="model-meta">'+(m.builtin?'Integriert':m.size_mb+' MB')+' · '+esc(m.format)+'</div>';
    if(isDl){html+='<div class="dl-bar"><div class="dl-bar-fill" style="width:'+prog.pct+'%"></div></div><div style="font-size:.66rem;color:rgba(255,255,255,.3);margin-top:3px">'+ prog.pct+'% heruntergeladen…</div>';}
    if(prog&&prog.status==='error')html+='<div style="font-size:.68rem;color:#ff453a;margin-top:4px">⚠ '+esc(prog.error)+'</div>';
    html+='</div><div class="model-actions">';
    if(isDl)html+='<button class="m-btn m-btn-install" disabled>Lädt…</button>';
    else if(!m.builtin&&!isInstalled&&m.url)html+='<button class="m-btn m-btn-install" data-action="install" data-id="'+esc(m.id)+'" data-url="'+esc(m.url)+'">Installieren</button>';
    if((isInstalled||m.builtin)&&!isActive)html+='<button class="m-btn m-btn-activate" data-action="activate" data-id="'+esc(m.id)+'" data-type="'+type+'">Aktivieren</button>';
    if(isInstalled&&!m.builtin)html+='<button class="m-btn m-btn-delete" data-action="delete" data-id="'+esc(m.id)+'">Löschen</button>';
    html+='</div></div>';
  });
  document.getElementById(listId).innerHTML=html;
}
async function settingsRefreshCatalog(){
  var btn=document.getElementById('refreshCatalogBtn');btn.disabled=true;btn.textContent='Aktualisiert…';
  try{await fetch('/api/models/refresh',{method:'POST'});await loadSettings();}catch(e){}
  btn.disabled=false;btn.textContent='↻ Aktualisieren';
}
async function installModel(id,url){
  try{
    await fetch('/api/models/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id,url:url,hf_token:hfToken})});
    var iv=setInterval(async function(){
      await loadSettings();
      if(settingsData&&settingsData.install_progress[id]){var s=settingsData.install_progress[id].status;if(s==='done'||s==='error')clearInterval(iv);}
    },800);
  }catch(e){}
}
async function activateModel(id,type){
  try{await fetch('/api/models/activate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id,type:type})});await loadSettings();}catch(e){}
}
async function deleteModel(id){
  if(!confirm('Modell wirklich löschen?'))return;
  try{await fetch('/api/models/'+id,{method:'DELETE'});await loadSettings();}catch(e){}
}
async function loadConfig(){
  try{
    var r=await fetch('/api/config');configData=await r.json();
    var fs=document.getElementById('frameSkipSlider'),cs=document.getElementById('confSlider');
    var fv=document.getElementById('frameSkipVal'),cv=document.getElementById('confVal');
    if(fs)fs.value=configData.detection_interval||4;
    if(fv)fv.textContent=configData.detection_interval||4;
    if(cs)cs.value=configData.plate_conf_thresh||0.45;
    if(cv)cv.textContent=(configData.plate_conf_thresh||0.45).toFixed(2);
  }catch(e){}
}
async function saveConfig(key,val){
  try{var body={};body[key]=val;await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}catch(e){}
}
function handleModelClick(e){
  var btn=e.target.closest('button[data-action]');if(!btn)return;
  var action=btn.dataset.action,id=btn.dataset.id;
  if(action==='install')installModel(id,btn.dataset.url);
  else if(action==='activate')activateModel(id,btn.dataset.type);
  else if(action==='delete')deleteModel(id);
}
async function refresh(){
  try{
    var r=await fetch('/status'),d=await r.json();
    var sdot=document.getElementById('sdot'),pillText=document.getElementById('pillText'),cancelBtn=document.getElementById('cancelBtn');
    document.getElementById('errBanner').style.display=d.error?'flex':'none';
    if(d.error)document.getElementById('errText').textContent=d.error;
    setChip('chipNvdec',d.hw_nvdec);setChip('chipNvenc',d.hw_nvenc);setChip('chipTrt',d.hw_trt);
    if(d.state==='idle'){
      sdot.className='status-dot'+(d.error?' error':'');pillText.textContent=d.error?'Fehler':'Bereit';
      cancelBtn.style.display='none';
      document.getElementById('jobCard').style.display='none';document.getElementById('renderCard').style.display='none';
      lastF=0;lastFT=Date.now();smoothFps=0;fpsHistory=[];
    }else if(d.state==='blur'){
      sdot.className='status-dot active pulse';pillText.textContent='Verarbeitet';
      cancelBtn.style.display='';cancelBtn.disabled=false;cancelBtn.textContent='Abbrechen';
      document.getElementById('jobCard').style.display='block';document.getElementById('renderCard').style.display='none';
      document.getElementById('jobName').textContent=d.name||'–';
      document.getElementById('jobSub').textContent='Video '+d.current+' von '+d.total;
      var pct=d.frame_pct||0;
      document.getElementById('pBar').style.width=pct+'%';
      document.getElementById('pLeft').textContent=pct+'% · '+(d.frame_current||0).toLocaleString('de-DE')+' / '+(d.frame_total||0).toLocaleString('de-DE')+' Frames';
      var fpsStr=calcFps(d.frame_current||0);
      document.getElementById('sFps').textContent=fpsStr;
      document.getElementById('sElapsed').textContent=fmt(d.elapsed_seconds);
      document.getElementById('sEta').textContent=fmt(d.eta_seconds);
      updateSparkline(smoothFps>0?smoothFps:0);
      document.getElementById('faceCount').textContent=(d.face_count||0).toLocaleString('de-DE');
      document.getElementById('plateCount').textContent=(d.plate_count||0).toLocaleString('de-DE');
    }else if(d.state==='render'){
      sdot.className='status-dot active pulse';pillText.textContent='Encodiert';
      cancelBtn.style.display='none';
      document.getElementById('jobCard').style.display='none';document.getElementById('renderCard').style.display='block';
      document.getElementById('renderName').textContent=d.out_name||'';
    }
    var box=document.getElementById('logBox'),atBottom=box.scrollHeight-box.clientHeight<=box.scrollTop+32;
    box.innerHTML=d.logs.map(function(l){return '<div class="le '+logClass(l)+'">'+renderLog(l)+'</div>';}).join('');
    if(autoScroll&&(atBottom||d.state!=='idle'))box.scrollTop=box.scrollHeight;
    document.getElementById('logCount').textContent=d.logs.length+' Einträge';
    document.getElementById('refreshTs').textContent='Aktualisiert: '+new Date().toLocaleTimeString('de-DE');
  }catch(e){
    document.getElementById('sdot').className='status-dot error';
    document.getElementById('pillText').textContent='Verbindung verloren';
  }
}
document.getElementById('frameSkipSlider').addEventListener('input',function(){var v=parseInt(this.value);document.getElementById('frameSkipVal').textContent=v;saveConfig('detection_interval',v);});
document.getElementById('confSlider').addEventListener('input',function(){var v=parseFloat(this.value);document.getElementById('confVal').textContent=v.toFixed(2);saveConfig('plate_conf_thresh',v);});
document.getElementById('gearBtn').addEventListener('click',openSettings);
document.getElementById('drawerClose').addEventListener('click',closeSettings);
document.getElementById('scrim').addEventListener('click',closeSettings);
document.getElementById('asBtn').addEventListener('click',toggleAs);
document.getElementById('cancelBtn').addEventListener('click',doCancel);
document.getElementById('hfTokenSaveBtn').addEventListener('click',saveHfToken);
document.getElementById('tabFace').addEventListener('click',function(){switchTab('face');});
document.getElementById('tabPlate').addEventListener('click',function(){switchTab('plate');});
document.getElementById('refreshCatalogBtn').addEventListener('click',settingsRefreshCatalog);
document.getElementById('faceModelList').addEventListener('click',handleModelClick);
document.getElementById('plateModelList').addEventListener('click',handleModelClick);
document.getElementById('hfTokenInput').value=hfToken;
loadConfig();loadSettings();refresh();setInterval(refresh,2000);
</script>
</body>
</html>"""
