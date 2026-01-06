// Initialize Mermaid for diagrams
document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: document.body.getAttribute('data-md-color-scheme') === 'slate' ? 'dark' : 'neutral',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
    }
});

// Re-render mermaid diagrams when theme changes
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.attributeName === 'data-md-color-scheme') {
            const scheme = document.body.getAttribute('data-md-color-scheme');
            if (typeof mermaid !== 'undefined') {
                mermaid.initialize({
                    theme: scheme === 'slate' ? 'dark' : 'neutral'
                });
                // Re-render all mermaid diagrams
                document.querySelectorAll('.mermaid').forEach(function(el) {
                    const code = el.getAttribute('data-original') || el.textContent;
                    el.setAttribute('data-original', code);
                    el.removeAttribute('data-processed');
                    el.innerHTML = code;
                });
                mermaid.init(undefined, '.mermaid');
            }
        }
    });
});

observer.observe(document.body, { attributes: true });
