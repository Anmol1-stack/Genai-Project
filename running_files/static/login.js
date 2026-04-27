document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('login-form');
    const loginBtn = document.getElementById('login-btn');
    const errorMsg = document.getElementById('login-error');

    // --- Toast Helper ---
    function showToast(message, isError = false) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.classList.remove('show', 'error');
        if (isError) toast.classList.add('error');
        void toast.offsetWidth;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 4000);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        loginBtn.textContent = 'Logging in...';
        loginBtn.disabled = true;
        errorMsg.style.display = 'none';

        const payload = {
            username: document.getElementById('username').value.trim(),
            password: document.getElementById('password').value
        };

        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (res.ok) {
                const data = await res.json();
                localStorage.setItem('hospital_name', data.hospital_name);
                showToast(`✅ Welcome, ${data.hospital_name}! Redirecting...`);
                setTimeout(() => { window.location.href = '/hospital-dashboard'; }, 1200);
            } else {
                const errorData = await res.json().catch(() => ({}));
                const msg = errorData.detail || "Invalid credentials. Please try again.";
                errorMsg.textContent = msg;
                errorMsg.style.display = 'block';
                showToast(`❌ ${msg}`, true);
            }
        } catch (error) {
            console.error("Error logging in", error);
            const msg = "Server error. Please try again later.";
            errorMsg.textContent = msg;
            errorMsg.style.display = 'block';
            showToast(`❌ ${msg}`, true);
        } finally {
            loginBtn.textContent = 'Login';
            loginBtn.disabled = false;
        }
    });
});
