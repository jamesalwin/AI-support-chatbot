// static/script.js
const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const template = document.getElementById("message-template").content;

function addMessage(text, isUser=false) {
  const el = template.cloneNode(true);
  const row = el.querySelector(".message-row");
  const bubble = el.querySelector(".bubble");
  const avatar = el.querySelector(".avatar");
  bubble.textContent = text;
  if (isUser) {
    row.classList.add("user");
    avatar.classList.add("bot"); // swap style or hide avatar for user if desired
    avatar.style.display = "none";
  } else {
    avatar.classList.add("bot");
  }
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "message-row";
  el.id = "typingRow";
  el.innerHTML = `<div class="avatar bot"></div>
    <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById("typingRow");
  if (t) t.remove();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, true);
  input.value = "";
  addTyping();
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();
    if (data && data.success) {
      // small delay to mimic typing
      setTimeout(() => addMessage(data.response, false), 300);
    } else {
      addMessage("Sorry, something went wrong. Try again later.", false);
    }
  } catch (err) {
    removeTyping();
    addMessage("Network error. Check the server.", false);
    console.error(err);
  }
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
