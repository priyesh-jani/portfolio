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

        // Check if the response is okay
        if (response.ok) {
            const data = await response.json();
            // Check if the answer field exists in the response
            if (data.answer) {
                chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
            } else {
                chatBox.innerHTML += `<p><strong>Bot:</strong> Sorry, I couldn't find an answer.</p>`;
            }
        } else {
            chatBox.innerHTML += `<p><strong>Bot:</strong> Sorry, there was an issue with your request.</p>`;
        }
    } catch (error) {
        chatBox.innerHTML += `<p><strong>Bot:</strong> Sorry, I couldn't connect to the server.</p>`;
    }

    // Scroll to the bottom of the chat box and clear the input
    chatBox.scrollTop = chatBox.scrollHeight;
    document.getElementById("user-input").value = "";
}
