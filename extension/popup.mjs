const DEFAULT_SETTINGS = {
  enabled: true,
  perfMode: "balanced",
};

function getElements() {
  return {
    enabledToggle: document.getElementById("enabledToggle"),
    modeSelect: document.getElementById("modeSelect"),
    rescanBtn: document.getElementById("rescanBtn"),
    statusEl: document.getElementById("status"),
  };
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

function init() {
  const { enabledToggle, modeSelect, rescanBtn, statusEl } = getElements();

  async function loadSettings() {
    const storage = await chrome.storage.local.get(DEFAULT_SETTINGS);
    enabledToggle.checked = storage.enabled;
    modeSelect.value = storage.perfMode;
    statusEl.textContent = "Settings loaded.";
  }

  async function saveSettings() {
    const payload = {
      enabled: enabledToggle.checked,
      perfMode: modeSelect.value,
    };
    await chrome.storage.local.set(payload);
    chrome.runtime.sendMessage({ type: "SETTINGS_UPDATED", payload });
    statusEl.textContent = "Settings saved.";
  }

  async function rescanNow() {
    const tab = await getActiveTab();
    if (!tab?.id) {
      statusEl.textContent = "No active tab found.";
      return;
    }
    chrome.tabs.sendMessage(tab.id, { type: "RESCAN_MEDIA" });
    statusEl.textContent = "Rescan triggered.";
  }

  enabledToggle.addEventListener("change", () => {
    void saveSettings();
  });
  modeSelect.addEventListener("change", () => {
    void saveSettings();
  });
  rescanBtn.addEventListener("click", () => {
    void rescanNow();
  });

  loadSettings().catch((err) => {
    statusEl.textContent = `Failed to load settings: ${err.message}`;
  });
}

document.addEventListener("DOMContentLoaded", init);
