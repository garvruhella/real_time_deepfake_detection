const DEFAULT_SETTINGS = {
  enabled: true,
  perfMode: "balanced"
};

const perfConfig = {
  fast: { videoIntervalMs: 1500, imageBatchSize: 2 },
  balanced: { videoIntervalMs: 800, imageBatchSize: 4 },
  quality: { videoIntervalMs: 400, imageBatchSize: 6 }
};

chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(DEFAULT_SETTINGS);
  await chrome.storage.local.set(current);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "SETTINGS_UPDATED") {
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach((tab) => {
        if (tab.id) {
          chrome.tabs.sendMessage(tab.id, {
            type: "SETTINGS_BROADCAST",
            payload: message.payload
          });
        }
      });
    });
    return;
  }

  if (message?.type === "GET_RUNTIME_CONFIG") {
    chrome.storage.local.get(DEFAULT_SETTINGS).then((settings) => {
      sendResponse({
        settings,
        perf: perfConfig[settings.perfMode] || perfConfig.balanced
      });
    });
    return true;
  }

  if (message?.type === "REPORT_DETECTION") {
    const tabId = sender?.tab?.id ?? "unknown";
    console.log("[DeepfakeDetector]", { tabId, ...message.payload });
  }
});
