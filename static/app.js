document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('complaint-form');
    const submitBtn = document.getElementById('submit-btn');
    const complaintList = document.getElementById('complaint-list');
    const filterAllBtn = document.getElementById('filter-all');
    const filterUrgentBtn = document.getElementById('filter-urgent');
    
    let currentFilter = 'all';

    // Fetch initial complaints
    fetchComplaints();

    // Filters
    filterAllBtn.addEventListener('click', () => {
        currentFilter = 'all';
        filterAllBtn.classList.add('active');
        filterUrgentBtn.classList.remove('active');
        fetchComplaints();
    });

    filterUrgentBtn.addEventListener('click', () => {
        currentFilter = 'urgent';
        filterUrgentBtn.classList.add('active');
        filterAllBtn.classList.remove('active');
        fetchComplaints();
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        submitBtn.textContent = 'Processing...';
        submitBtn.disabled = true;

        const payload = {
            image_caption: document.getElementById('image_caption').value,
            voice_text: document.getElementById('voice_text').value,
            hospital_name: document.getElementById('hospital_name').value,
            city: document.getElementById('city').value
        };

        try {
            const res = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (res.ok) {
                // Clear form
                form.reset();
                document.getElementById('hospital_name').value = "General Hospital";
                document.getElementById('city').value = "Metropolis";
                // Refresh list
                fetchComplaints();
            }
        } catch (error) {
            console.error("Error processing complaint", error);
            alert("Error processing complaint");
        } finally {
            submitBtn.textContent = 'Process Complaint';
            submitBtn.disabled = false;
        }
    });

    async function fetchComplaints() {
        const endpoint = currentFilter === 'all' ? '/complaints' : '/urgent';
        complaintList.innerHTML = '<p style="color: var(--text-muted); text-align: center;">Loading...</p>';
        
        try {
            const res = await fetch(endpoint);
            const data = await res.json();
            renderComplaints(data);
        } catch (error) {
            console.error("Error fetching complaints", error);
            complaintList.innerHTML = '<p style="color: red;">Failed to load complaints.</p>';
        }
    }

    function renderComplaints(complaints) {
        complaintList.innerHTML = '';
        if (complaints.length === 0) {
            complaintList.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No complaints found.</p>';
            return;
        }

        complaints.forEach(c => {
            const date = new Date(c.timestamp).toLocaleString();
            const badgeClass = c.severity.toLowerCase();
            
            const card = document.createElement('div');
            card.className = `complaint-card`;
            card.style.borderLeftColor = `var(--${badgeClass}-color)`;
            
            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title">${c.category} &bull; ${c.department}</div>
                    <span class="badge ${badgeClass}">${c.severity}</span>
                </div>
                <div class="card-meta">
                    Reported at ${date} &bull; ${c.hospital_name}, ${c.city}
                </div>
                <div class="card-desc">
                    ${c.description}
                </div>
                ${c.relevant_sop ? `
                <div class="card-sop">
                    <strong>SOP Match:</strong> 
                    <span>${c.relevant_sop}</span>
                </div>` : ''}
            `;
            complaintList.appendChild(card);
        });
    }
});
