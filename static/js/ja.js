            function updateSlider(value) {
            document.getElementById("kernel-value").textContent = value;
            updateImage();
        }

        function updateImage() {
            const imageName = document.querySelector('input[name="image_name"]:checked').value;
            const kernelSize = document.getElementById("kernel-slider").value;
            const url = `/partb?image_name=${imageName}&kernel_size=${kernelSize}`;
            
            // 1. Obtener el contenedor a actualizar
            const container = document.getElementById("processed-images-container"); 

            // Actualizar la imagen y operaciones sin recargar la página
            fetch(url)
                .then(response => response.text())
                .then(html => {
                    // Crea un elemento temporal para parsear el HTML completo
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');

                    // Extrae solo el contenido del contenedor de resultados (el #processed-images-container)
                    const newContent = doc.getElementById('processed-images-container').innerHTML;

                    // Reemplaza el contenido del contenedor actual
                    container.innerHTML = newContent; 
                });
        }