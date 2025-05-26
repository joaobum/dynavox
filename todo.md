1. I don't want the HTTP calls logged as in:
INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK", instead log just the request prompt and response text at DEBUG level.
2. Creage a logs folder and each run has logs pushed to that folder, with a filanem that has run info and timestamp, similar to results.
3. Add to both logs and results filemanes or folders a suffix _mock in case it's mock
