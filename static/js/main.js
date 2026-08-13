const fileInput = document.getElementById('file-upload');
const dropzoneLabel = document.getElementById('dropzone-label');
const dropzoneText = document.getElementById('dropzone-text');
const dropzoneHint = document.getElementById('dropzone-hint');

if (fileInput) {
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file) return;
        dropzoneLabel.classList.add('has-file');
        dropzoneText.textContent = file.name;
        dropzoneHint.textContent = 'Ready to analyze';
    });
}
