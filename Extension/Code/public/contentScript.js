async function setSyncStorage(data) {
    return new Promise((resolve, reject) => {
        chrome.storage.sync.set(data, () => {
            if (chrome.runtime.lastError) {
                reject("Error setting data to sync storage");
            } else {
                resolve();
            }
        });
    });
}

async function handleTextSelection() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
  
    if (selectedText) {
      console.log('Selected text:', selectedText);
      await setSyncStorage({ text: selectedText });
    }
}
  
document.addEventListener('mouseup', handleTextSelection);