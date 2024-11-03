document.getElementById("send-button").addEventListener("click", sendQuestion);

async function sendQuestion() {
    const question = document.getElementById("user-input").value;
    if (question.trim() === "") return;

    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p><strong>You:</strong> ${question}</p>`;

    try {
        const response = await fetch("https://<your-replit-username>.repl.co/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
    } catch (error) {
        chatBox.innerHTML += `<p><strong>Bot:</strong> Sorry, I couldn't connect to the server.</p>`;
    }

    chatBox.scrollTop = chatBox.scrollHeight;
    document.getElementById("user-input").value = "";
}
