const text = `**1. Descripción de la imagen del usuario**
La imagen de consulta muestra una interfaz.

**2. Características histológicas según el manual**
El manual describe la "Histología de tejido muscular" [Manual: arch2.pdf, Sección 3].

*   La morfología de las células fusiformes.
*   En contraste, el tejido conectivo denso.
*   El tejido conectivo laxo es menos organizado.

**4. Diagnóstico diferencial con diferencias morfológicas clave**
*   **Músculo liso:** Se caracteriza por células.
*   **Tejido conectivo denso/fibroso denso:** Predominan las fibras.
*   **Tejido conectivo laxo/Estroma reactivo:** Presenta una matriz.

**5. Conclusión y confianza**
Confianza: 5/5.`;

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold + Italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Lists
    html = html.replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    // Remove nested <ul> wraps
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Numbered lists
    html = html.replace(/^\s*\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already structured
    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

console.log(renderMarkdown(text));
