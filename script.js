document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput) {
        const userMessage = document.createElement('p');
        userMessage.textContent = "You: " + userInput;
        document.getElementById('chat-box').appendChild(userMessage);
        
        fetch('https://<your-replit-url>/ask', {  // Replace with your Replit URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: userInput })
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('p');
            botMessage.textContent = "Chatbot: " + data.answer;
            document.getElementById('chat-box').appendChild(botMessage);
            document.getElementById('user-input').value = '';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

