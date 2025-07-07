document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const chatSection = document.getElementById('chat-section');
    const chatWindow = document.getElementById('chat-window');
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        const fileInput = document.getElementById('file-input');
        if (!fileInput.files || fileInput.files.length === 0) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'red';
            return;
        }
        uploadStatus.textContent = 'Uploading and processing... This may take a moment.';
        uploadStatus.style.color = 'blue';
        chatWindow.innerHTML = ''; // Clear previous chat
        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok) {
                uploadStatus.textContent = result.message;
                uploadStatus.style.color = 'green';
                chatSection.classList.remove('hidden');
                addMessageToChat(result.initial_analysis, 'bot');
            } else { throw new Error(result.error || 'Unknown error'); }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.style.color = 'red';
        }
    });

    const handleAskQuestion = async () => {
        const question = questionInput.value.trim();
        if (!question) return;
        addMessageToChat(question, 'user');
        questionInput.value = '';
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });
            const result = await response.json();
            if (response.ok) { addMessageToChat(result.answer, 'bot'); } 
            else { throw new Error(result.error || 'Unknown error'); }
        } catch (error) { addMessageToChat(`Error: ${error.message}`, 'bot'); }
    };

    askButton.addEventListener('click', handleAskQuestion);
    questionInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') { handleAskQuestion(); }});

    function addMessageToChat(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = text;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});