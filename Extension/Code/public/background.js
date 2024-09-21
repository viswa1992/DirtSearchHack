chrome.contextMenus.create({
    id: "factCheck",
    title: "Reality check",
    contexts: ["selection"] // Show this menu item when text is selected
});

chrome.contextMenus.onClicked.addListener(function(info, tab) { 
    if (info.menuItemId === "factCheck") {
        chrome.action.openPopup();
    }
});