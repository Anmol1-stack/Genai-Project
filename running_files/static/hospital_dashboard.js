document.addEventListener('DOMContentLoaded', () => {
    const hospitalName = localStorage.getItem('hospital_name');
    
    if (!hospitalName) {
        // Not logged in, redirect
        window.location.href = '/login';
        return;
    }

    document.getElementById('hospital-title').textContent = `${hospitalName} Dashboard`;

    const complaintList = document.getElementById('complaint-list');
    const filterAllBtn = document.getElementById('filter-all');
    const filterUrgentBtn = document.getElementById('filter-urgent');
    const logoutBtn = document.getElementById('logout-btn');
    
    let currentFilter = 'all';

    fetchComplaints();

    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('hospital_name');
        window.location.href = '/login';
    });

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

    async function fetchComplaints() {
        const baseEndpoint = currentFilter === 'all' ? '/complaints' : '/urgent';
        // Add hospital_name as a query parameter
        const endpoint = `${baseEndpoint}?hospital_name=${encodeURIComponent(hospitalName)}`;
        
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
            complaintList.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No complaints found for your hospital.</p>';
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
                    Reported at ${date} &bull; ${c.city}
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
