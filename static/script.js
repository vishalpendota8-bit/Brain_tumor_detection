// Shows a preview of the chosen image before uploading.
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const uploadText = document.getElementById("uploadText");

fileInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.hidden = false;
            uploadText.textContent = file.name;
        };
        reader.readAsDataURL(file);
    }
});
