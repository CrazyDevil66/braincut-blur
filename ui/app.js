var lastF=0,lastFT=Date.now(),smoothFps=0,autoScroll=true;
var settingsData=null,activeTab='face',hfToken=localStorage.getItem('hf_token')||''
var fpsHistory=[],configData={detection_interval:4,plate_conf_thresh:0.45,face_conf_thresh:0.2};
var sbVisible=false,sbHideTimer=null;
var previewTimer=null,previewActive=false;

function startPreview(){
  if(previewActive)return;
  previewActive=true;
  document.getElementById('previewCard').style.display='block';
  function poll(){
    var img=document.getElementById('previewImg');
    if(img)img.src='/api/frame?t='+Date.now();
    previewTimer=setTimeout(poll,1500);
  }
  poll();
}
function stopPreview(){
  previewActive=false;
  if(previewTimer){clearTimeout(previewTimer);previewTimer=null;}
  document.getElementById('previewCard').style.display='none';
  var img=document.getElementById('previewImg');
  if(img)img.src='';
}
var SB_KEYS=['probe','load_models','blur_loop','encode','mux'];

function renderStepsBar(activeIdx,isErr){
  for(var i=0;i<6;i++){
    var c=document.getElementById('sc'+i);
    var lbl=document.getElementById('slbl'+i);
    if(!c||!lbl)continue;
    c.className='step-circle';c.textContent='';lbl.className='step-lbl';
    var lineR=document.getElementById('sl'+i);
    var lineL=document.getElementById('sl'+(i-1)+'r');
    if(lineR)lineR.className='step-line'+(i<activeIdx?' done':'');
    if(lineL)lineL.className='step-line'+(i<=activeIdx?' done':'');
    if(i<activeIdx){c.className='step-circle done';c.textContent='✓';lbl.className='step-lbl done';}
    else if(i===activeIdx){
      if(isErr){c.className='step-circle err';c.textContent='✕';lbl.className='step-lbl err';}
      else{c.className='step-circle active pulse';lbl.className='step-lbl active';}
    }
  }
}

function updateStepsBar(d){
  var bar=document.getElementById('stepsBar');
  var active=SB_KEYS.indexOf(d.sub_state||'');
  var isErr=d.state==='error'||d.state==='cancelled';
  if(d.state==='blur'||d.state==='render'){
    sbVisible=true;
    if(sbHideTimer){clearTimeout(sbHideTimer);sbHideTimer=null;}
    bar.style.display='block';
    renderStepsBar(active,false);
  }else if(d.state==='idle'&&sbVisible){
    if(!sbHideTimer){
      renderStepsBar(5,false);
      sbHideTimer=setTimeout(function(){bar.style.display='none';sbVisible=false;sbHideTimer=null;},3000);
    }
  }else if(isErr&&sbVisible){
    renderStepsBar(active,true);
  }else if(!sbVisible){
    bar.style.display='none';
  }
}

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
  var html='';
  // Kennzeichen: "Kein Blur"-Option am Anfang
  if(type==='plate'){
    var noneActive=!activeId||activeId==='null'||activeId===null;
    html+='<div class="model-row'+(noneActive?' active':'')+'"><div class="model-icon pi" style="font-size:.9rem;color:rgba(255,255,255,.25)">&#8722;</div>';
    html+='<div class="model-info"><div class="model-name">Kein Kennzeichen-Blur'+(noneActive?'<span class="active-badge">Aktiv</span>':'')+'</div>';
    html+='<div class="model-desc">Kennzeichen werden nicht verblurrt</div></div>';
    html+='<div class="model-actions">';
    if(!noneActive)html+='<button class="m-btn m-btn-delete" style="background:rgba(255,159,10,.1);color:#ff9f0a" data-action="activate-none" data-type="plate">Deaktivieren</button>';
    html+='</div></div>';
  }
  if(!models.length){document.getElementById(listId).innerHTML=html+'<div style="color:rgba(255,255,255,.3);font-size:.8rem;padding:8px 0">Keine Modelle im Katalog.</div>';return;}
  models.forEach(function(m){
    var isActive=m.id===activeId,isInstalled=!!(d.installed[m.id]),prog=d.install_progress[m.id],isDl=prog&&prog.status==='downloading';
    html+='<div class="model-row'+(isActive?' active':'')+'"><div class="model-icon '+iconCls+'">'+ iconChar +'</div>';
    html+='<div class="model-info"><div class="model-name">'+esc(m.name);
    if(isActive)html+='<span class="active-badge">Aktiv</span>';
    html+='</div><div class="model-desc">'+esc(m.description)+'</div>';
    html+='<div class="model-meta">'+(m.builtin?'Integriert':m.size_mb+' MB')+(m.format?' · '+esc(m.format):'')+'</div>';
    if(isDl){html+='<div class="dl-bar"><div class="dl-bar-fill" style="width:'+prog.pct+'%"></div></div><div style="font-size:.66rem;color:rgba(255,255,255,.3);margin-top:3px">'+ prog.pct+'% heruntergeladen…</div>';}
    if(prog&&prog.status==='error')html+='<div style="font-size:.68rem;color:#ff453a;margin-top:4px">⚠ '+esc(prog.error)+'</div>';
    html+='</div><div class="model-actions">';
    if(isDl)html+='<button class="m-btn m-btn-install" disabled>Lädt…</button>';
    else if(!m.builtin&&!isInstalled)html+='<button class="m-btn m-btn-install" data-action="install" data-id="'+esc(m.id)+'" data-url="'+esc(m.url||'')+'">Installieren…</button>';
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
async function installModel(id,suggestedUrl){
  var url=window.prompt('Download-URL für '+id+':\n(HuggingFace-Token oben eintragen falls nötig)',suggestedUrl||'');
  if(!url||!url.trim())return;
  try{
    await fetch('/api/models/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id,url:url.trim(),hf_token:hfToken})});
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
    var fcs=document.getElementById('faceConfSlider'),fcv=document.getElementById('faceConfVal');
    if(fs)fs.value=configData.detection_interval||4;
    if(fv)fv.textContent=configData.detection_interval||4;
    if(cs)cs.value=configData.plate_conf_thresh||0.45;
    if(cv)cv.textContent=(configData.plate_conf_thresh||0.45).toFixed(2);
    if(fcs)fcs.value=configData.face_conf_thresh||0.2;
    if(fcv)fcv.textContent=(configData.face_conf_thresh||0.2).toFixed(2);
    var fct=document.getElementById('faceComboToggle');
    if(fct)fct.checked=configData.face_combo!==false;
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
  else if(action==='activate-none')activateModel(null,btn.dataset.type);
  else if(action==='delete')deleteModel(id);
}
async function refresh(){
  try{
    var r=await fetch('/status'),d=await r.json();
    var sdot=document.getElementById('sdot'),pillText=document.getElementById('pillText'),cancelBtn=document.getElementById('cancelBtn');
    document.getElementById('errBanner').style.display=d.error?'flex':'none';
    if(d.error)document.getElementById('errText').textContent=d.error;
    setChip('chipNvdec',d.hw_nvdec);setChip('chipNvenc',d.hw_nvenc);setChip('chipTrt',d.hw_trt);
    updateStepsBar(d);
    if(d.state==='idle'){
      sdot.className='status-dot'+(d.error?' error':'');pillText.textContent=d.error?'Fehler':'Bereit';
      cancelBtn.style.display='none';
      document.getElementById('jobCard').style.display='none';document.getElementById('renderCard').style.display='none';
      stopPreview();
      lastF=0;lastFT=Date.now();smoothFps=0;fpsHistory=[];
    }else if(d.state==='blur'){
      sdot.className='status-dot active pulse';pillText.textContent='Verarbeitet';
      cancelBtn.style.display='';cancelBtn.disabled=false;cancelBtn.textContent='Abbrechen';
      document.getElementById('jobCard').style.display='block';document.getElementById('renderCard').style.display='none';
      startPreview();
      document.getElementById('jobName').textContent=d.name||'–';
      document.getElementById('jobSub').textContent='Video '+d.current+' von '+d.total;
      var pct=d.frame_pct||0;
      document.getElementById('pBar').style.width=pct+'%';
      document.getElementById('pLeft').textContent=pct+'% · '+(d.frame_current||0).toLocaleString('de-DE')+' / '+(d.frame_total||0).toLocaleString('de-DE')+' Frames';
      document.getElementById('pRight').textContent=d.eta_seconds>0?'~'+fmt(d.eta_seconds)+' verbleibend':'';
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
document.getElementById('faceConfSlider').addEventListener('input',function(){var v=parseFloat(this.value);document.getElementById('faceConfVal').textContent=v.toFixed(2);saveConfig('face_conf_thresh',v);});
document.getElementById('faceComboToggle').addEventListener('change',function(){saveConfig('face_combo',this.checked);});
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
