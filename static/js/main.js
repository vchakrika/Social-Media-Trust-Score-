'use strict';

// ── State ────────────────────────────────────────────────
let currentPlatform = 'instagram';
let history = JSON.parse(localStorage.getItem('trustscore_history') || '[]');

// ── Platform Switch ──────────────────────────────────────
function switchPlatform(p) {
  currentPlatform = p;
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.platform === p);
  });
  document.getElementById('instagram-form').style.display = p === 'instagram' ? '' : 'none';
  document.getElementById('facebook-form').style.display  = p === 'facebook'  ? '' : 'none';
  document.getElementById('form-subtitle').textContent =
    p === 'instagram'
      ? "Enter the Instagram account's public metrics below"
      : "Enter the Facebook account's public metrics below";
  renderImportanceBars(p);
  document.getElementById('importance-subtitle').textContent =
    `Top factors used by the ${p.charAt(0).toUpperCase() + p.slice(1)} AI model`;
}

// ── Toggle Buttons ────────────────────────────────────────
function setToggle(btn, fieldId) {
  const group = btn.closest('.toggle-group');
  group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(fieldId).value = btn.dataset.val;
}

// ── Slider Sync ───────────────────────────────────────────
function syncSlider(targetId, value) {
  const el = document.getElementById(targetId);
  if (el) el.value = value;
}

// ── Collect Form Data ─────────────────────────────────────
function getFormData() {
  const g = id => parseFloat(document.getElementById(id)?.value || 0) || 0;
  if (currentPlatform === 'instagram') {
    return {
      platform: 'instagram',
      profile_pic: g('profile_pic'),
      nums_length_username: g('nums_length_username'),
      fullname_words: g('fullname_words'),
      nums_length_fullname: g('nums_length_fullname'),
      name_eq_username: g('name_eq_username'),
      description_length: g('description_length'),
      external_url: g('external_url'),
      private: g('private'),
      posts: g('posts'),
      followers: g('followers'),
      following: g('following'),
    };
  } else {
    return {
      platform: 'facebook',
      friends: g('friends'),
      following: g('fb_following'),
      community: g('community'),
      age: g('age'),
      postshared: g('postshared'),
      urlshared: g('urlshared'),
      photos_videos: g('photos_videos'),
      fpurls: g('fpurls'),
      fpphotos: g('fpphotos'),
      avgcomment: g('avgcomment'),
      likes_post: g('likes_post'),
      tags_post: g('tags_post'),
      num_tags: g('num_tags'),
    };
  }
}

// ── Analyze ───────────────────────────────────────────────
async function analyze() {
  const btnText    = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  btnText.style.display    = 'none';
  btnSpinner.style.display = 'inline';

  try {
    const payload = getFormData();
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderResults(data);
    addToHistory(data);
  } catch (e) {
    alert('Error connecting to the server. Make sure Flask is running.');
  } finally {
    btnText.style.display    = 'inline';
    btnSpinner.style.display = 'none';
  }
}

// ── Render Results ────────────────────────────────────────
function renderResults(data) {
  document.getElementById('score-placeholder').style.display = 'none';
  document.getElementById('score-result').style.display = '';

  // Verdict badge
  const badge = document.getElementById('verdict-badge');
  badge.textContent = data.verdict;
  badge.className = `verdict-badge ${data.verdict_class}`;

  // Gauge arc
  const arc = document.getElementById('gauge-arc');
  const scoreText = document.getElementById('gauge-score-text');
  const totalLen = 267;
  const filled = (data.trust_score / 100) * totalLen;
  arc.style.strokeDashoffset = totalLen - filled;
  scoreText.textContent = data.trust_score;

  // Color the arc
  let arcColor;
  if (data.trust_score >= 75)      arcColor = '#22c55e';
  else if (data.trust_score >= 45) arcColor = '#f59e0b';
  else                             arcColor = '#ef4444';
  arc.style.stroke = arcColor;
  scoreText.style.fill = arcColor;

  // Probability bars
  document.getElementById('prob-real-pct').textContent = data.prob_real + '%';
  document.getElementById('prob-fake-pct').textContent = data.prob_fake + '%';
  document.getElementById('prob-real-bar').style.width = data.prob_real + '%';
  document.getElementById('prob-fake-bar').style.width = data.prob_fake + '%';

  // Risk flags
  const riskCard = document.getElementById('risk-card');
  const flagsList = document.getElementById('risk-flags-list');
  if (data.risk_flags && data.risk_flags.length > 0) {
    riskCard.style.display = '';
    flagsList.innerHTML = data.risk_flags.map(f => `
      <div class="risk-flag ${f.severity}">
        <span style="font-size:1.1rem">${f.icon}</span>
        <span class="risk-flag-text">${f.text}</span>
      </div>
    `).join('');
  } else {
    riskCard.style.display = '';
    flagsList.innerHTML = `<div class="risk-flag low"><span>✅</span><span class="risk-flag-text">No significant risk flags detected.</span></div>`;
  }

  // Scroll to results
  document.getElementById('score-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Feature Importance Bars ───────────────────────────────
function renderImportanceBars(platform) {
  const container = document.getElementById('importance-bars');
  if (!container) return;
  const importances = platform === 'instagram' ? IG_IMPORTANCES : FB_IMPORTANCES;
  const top8 = importances.slice(0, 8);
  const max = top8[0][1];

  const labels = {
    'profile pic': 'Profile Picture', 'nums/length username': 'Numbers in Username',
    'fullname words': 'Words in Full Name', 'nums/length fullname': 'Numbers in Full Name',
    'name==username': 'Name = Username', 'description length': 'Bio Length',
    'external URL': 'External URL', 'private': 'Private Account',
    '#posts': '# Posts', '#followers': '# Followers', '#follows': '# Following',
    '#friends': '# Friends', '#following': '# Following', '#community': '# Community',
    'age': 'Account Age', '#postshared': '# Posts Shared', '#urlshared': '# URLs Shared',
    '#photos/videos': '# Photos/Videos', 'fpurls': 'URL-Only Posts Ratio',
    'fpphotos/videos': 'Photo-Only Posts Ratio', 'avgcomment/post': 'Avg Comments/Post',
    'likes/post': 'Likes per Post', 'tags/post': 'Tags per Post', '#tags/post': '# Tags'
  };

  container.innerHTML = top8.map(([feat, val]) => {
    const pct = ((val / max) * 100).toFixed(1);
    const label = labels[feat] || feat;
    return `
      <div class="imp-bar-item">
        <div class="imp-bar-label"><span>${label}</span><span>${(val * 100).toFixed(1)}%</span></div>
        <div class="imp-bar-bg"><div class="imp-bar-fill" style="width:${pct}%"></div></div>
      </div>`;
  }).join('');
}

// ── History ───────────────────────────────────────────────
function addToHistory(data) {
  const entry = {
    trust_score: data.trust_score,
    verdict: data.verdict,
    verdict_class: data.verdict_class,
    platform: data.platform,
    prob_fake: data.prob_fake,
    time: new Date().toLocaleTimeString()
  };
  history.unshift(entry);
  if (history.length > 20) history.pop();
  localStorage.setItem('trustscore_history', JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById('history-list');
  if (!history.length) {
    el.innerHTML = '<p class="empty-history">No analyses yet.</p>';
    return;
  }
  el.innerHTML = `<div class="history-grid">${
    history.map(h => `
      <div class="history-item">
        <div class="history-score-circle ${h.verdict_class}">${h.trust_score}</div>
        <div class="history-info">
          <div class="history-verdict">${h.verdict}</div>
          <div class="history-platform">${h.platform.charAt(0).toUpperCase() + h.platform.slice(1)} · ${h.prob_fake}% fake probability · ${h.time}</div>
        </div>
      </div>`).join('')
  }</div>`;
}

function clearHistory() {
  history = [];
  localStorage.removeItem('trustscore_history');
  renderHistory();
}

// ── Init ──────────────────────────────────────────────────
renderHistory();
