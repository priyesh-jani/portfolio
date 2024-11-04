<script>
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to send a predefined question
    function sendPredefinedQuestion(question) {
        chatBox.innerHTML += `<div><strong>You:</strong> ${question}</div>`;
        userInput.value = ''; // Clear input field
        sendQuestion(question); // Call sendQuestion with the predefined question
    }

    // Function to send a question to the server and receive an answer
    async function sendQuestion(userQuestion) {
        if (userQuestion) {
            try {
                console.log('Sending question:', userQuestion); // Log the question being sent

                // Send request to Flask backend (update this URL for production)
                const response = await fetch(`https://portfolio-nqpj.onrender.com/get_response`, {
                    method: 'POST', 
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: userQuestion }) // Send question as JSON
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log('Received response:', data); // Log the response

                    if (data.answer) {
                        chatBox.innerHTML += `<div><strong>Chatbot:</strong> ${data.answer}</div>`;
                    } else {
                        chatBox.innerHTML += `<div><strong>Chatbot:</strong> Sorry, I couldn't understand that.</div>`;
                    }
                    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
                } else {
                    chatBox.innerHTML += `<div><strong>Chatbot:</strong> Sorry, I couldn't understand that.</div>`;
                }
            } catch (error) {
                console.error('Error:', error);
                chatBox.innerHTML += `<div><strong>Chatbot:</strong> An error occurred. Please try again.</div>`;
            }
        }
    }

    // Event listener for sending input
    sendButton.addEventListener('click', () => {
        const userQuestion = userInput.value.trim(); // Trim whitespace
        if (userQuestion) {
            chatBox.innerHTML += `<div><strong>You:</strong> ${userQuestion}</div>`;
            sendQuestion(userQuestion); // Call function to send question
            userInput.value = ''; // Clear input field
        }
    });

    // Allow pressing 'Enter' to send the question
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendButton.click(); // Trigger the click event of the send button
        }
    });
</script>
