document.addEventListener('DOMContentLoaded', () => {
    const complaintList  = document.getElementById('complaint-list');
    const filterAllBtn   = document.getElementById('filter-all');
    const filterUrgentBtn = document.getElementById('filter-urgent');
    const filterResolvedBtn = document.getElementById('filter-resolved');

    let currentFilter = 'all';

    fetchComplaints();

    filterAllBtn.addEventListener('click', () => {
        setFilter('all');
    });
    filterUrgentBtn.addEventListener('click', () => {
        setFilter('urgent');
    });
    filterResolvedBtn.addEventListener('click', () => {
        setFilter('resolved');
    });

    function setFilter(f) {
        currentFilter = f;
        [filterAllBtn, filterUrgentBtn, filterResolvedBtn].forEach(b => b.classList.remove('active'));
        document.getElementById(`filter-${f}`).classList.add('active');
        fetchComplaints();
    }

    // ── Toast helper ──────────────────────────────────────────────────────────
    function showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = '';
        if (type !== 'success') toast.classList.add(type);
        void toast.offsetWidth;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show', type), 4000);
    }

    // ── Fetch ─────────────────────────────────────────────────────────────────
    async function fetchComplaints() {
        let endpoint;
        if (currentFilter === 'urgent')   endpoint = '/urgent';
        else if (currentFilter === 'resolved') endpoint = '/complaints?status=resolved';
        else                              endpoint = '/complaints';

        complaintList.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">Loading...</p>';

        try {
            const res  = await fetch(endpoint);
            const data = await res.json();
            renderComplaints(data);
        } catch (err) {
            console.error(err);
            complaintList.innerHTML = '<p style="color:red;text-align:center;">Failed to load complaints.</p>';
        }
    }

    // ── Resolve action ────────────────────────────────────────────────────────
    async function resolveComplaint(id, cardEl) {
        const btn = cardEl.querySelector('.resolve-btn');
        btn.disabled = true;
        btn.textContent = 'Resolving…';

        try {
            const res = await fetch(`/complaints/${id}/resolve`, { method: 'PATCH' });
            if (!res.ok) throw new Error(await res.text());

            // Slide the card out then remove it
            cardEl.style.transition = 'opacity 0.4s ease, transform 0.4s ease, max-height 0.4s ease, margin 0.4s ease, padding 0.4s ease';
            cardEl.style.overflow   = 'hidden';
            cardEl.style.maxHeight  = cardEl.offsetHeight + 'px';
            requestAnimationFrame(() => {
                cardEl.style.opacity   = '0';
                cardEl.style.transform = 'translateX(60px)';
                cardEl.style.maxHeight = '0';
                cardEl.style.margin    = '0';
                cardEl.style.padding   = '0';
            });
            setTimeout(() => {
                cardEl.remove();
                showToast('Complaint marked as resolved ✓');
                // If list is now empty, show message
                if (!complaintList.querySelector('.complaint-card')) {
                    complaintList.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No active complaints.</p>';
                }
            }, 420);
        } catch (err) {
            console.error(err);
            btn.disabled    = false;
            btn.textContent = 'Resolve';
            showToast('Failed to resolve complaint', 'error');
        }
    }

    // ── Render ────────────────────────────────────────────────────────────────
    function renderComplaints(complaints) {
        complaintList.innerHTML = '';
        if (!complaints.length) {
            complaintList.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem;">No complaints found.</p>';
            return;
        }

        complaints.forEach(c => {
            const date       = new Date(c.timestamp).toLocaleString();
            const badgeClass = c.severity.toLowerCase();
            const isResolved = c.status === 'resolved';

            const card = document.createElement('div');
            card.className = 'complaint-card';
            card.style.borderLeftColor = `var(--${badgeClass}-color)`;
            if (isResolved) card.style.opacity = '0.65';

            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title">${c.category} &bull; ${c.department}</div>
                    <div style="display:flex;align-items:center;gap:0.6rem;">
                        <span class="badge ${badgeClass}">${c.severity}</span>
                        ${isResolved
                            ? `<span class="badge resolved-badge">Resolved</span>`
                            : `<button class="resolve-btn" data-id="${c.id}">&#10003; Resolve</button>`
                        }
                    </div>
                </div>
                <div class="card-meta">
                    Reported at ${date} &bull; ${c.hospital_name}, ${c.city}
                </div>
                <div class="card-desc">${c.description}</div>
                ${c.relevant_sop ? `
                <div class="card-sop">
                    <strong>SOP Match:</strong> <span>${c.relevant_sop}</span>
                </div>` : ''}
            `;

            // Wire up the resolve button
            const resolveBtn = card.querySelector('.resolve-btn');
            if (resolveBtn) {
                resolveBtn.addEventListener('click', () => resolveComplaint(c.id, card));
            }

            complaintList.appendChild(card);
        });
    }
});
