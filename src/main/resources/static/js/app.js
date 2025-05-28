        function predict(btn, path) {
            btn.disabled = true;
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'path=' + encodeURIComponent(path)
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("result-" + path.replace(/[\/\.]/g, '_')).innerText = data.prediction;
            })
            .catch(err => alert("Error predicting"))
            .finally(() => btn.disabled = false);
        }