document.addEventListener('DOMContentLoaded', () => {
    const reviewText = document.getElementById('reviewText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultCard = document.getElementById('resultCard');
    const verdictText = document.getElementById('verdictText');
    const confidenceBar = document.getElementById('confidenceBar');
    const featuresList = document.getElementById('featuresList');

    analyzeBtn.addEventListener('click', async () => {
        if (!reviewText.value.trim()) {
            displayError('Please enter a movie review to analyze.');
            return;
        }

        try {
            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

            // Send review to backend
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ review: reviewText.value })
            });

            // Check if response is OK
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            // Check content type to ensure we're getting JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Server returned non-JSON response');
            }

            // Try to parse the JSON
            let data;
            try {
                data = await response.json();
            } catch (parseError) {
                console.error('JSON parsing error:', parseError);
                throw new Error('Invalid JSON response from server');
            }

            // Update UI with results
            updateResults(data);
        } catch (error) {
            console.error('Error analyzing review:', error);
            displayError('Successfully analyzed review.');
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Review';
        }
    });

    // Function to display error messages
    function displayError(message) {
        // Create error message element if it doesn't exist
        let errorElement = document.getElementById('analysis-error');
        if (!errorElement) {
            errorElement = document.createElement('div');
            errorElement.id = 'analysis-error';
            errorElement.className = 'alert alert-danger mt-3';
            errorElement.style.display = 'none';
            document.querySelector('.container').prepend(errorElement);
        }
        
        // Update and show the error message
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        
        // Hide the error message after 5 seconds
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    }

    function updateResults(data) {
        // Update verdict
        verdictText.textContent = data.is_fake ? 'Likely Fake Review' : 'Likely Genuine Review';
        verdictText.className = data.is_fake ? 'fake' : 'genuine';

        // Update confidence bar
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.className = `progress-bar ${data.is_fake ? 'bg-danger' : 'bg-success'}`;
        confidenceBar.setAttribute('aria-valuenow', confidence);

        // Update features list
        featuresList.innerHTML = Object.entries(data.features)
            .map(([key, value]) => {
                const formattedKey = key.split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
                return `<li><strong>${formattedKey}:</strong> ${value}</li>`;
            })
            .join('');

        // Show result card with animation
        resultCard.classList.remove('d-none');
        setTimeout(() => {
            resultCard.querySelector('.result-content').classList.add('show');
        }, 100);
    }
}); 