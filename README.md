# AI Voice-to-Text Agent

A Streamlit voice transcription and chat application powered by Hugging Face Whisper Large v3, Llama 3.1 8B Instruct, and LangChain.

## Features

- Microphone recording, audio upload (WAV/MP3/M4A), and optional local sample audio.
- Whisper Large v3 transcription with Tamil and English language hints or automatic detection.
- Editable transcript, TXT download, and browser-safe copy workflow.
- Llama 3.1 8B chat grounded in the transcript, with session chat history and quick prompts.
- Purple/white responsive dashboard with a dark sidebar.

## Windows installation

1. Install Python 3.10 or 3.11 and ensure **Add Python to PATH** is selected.
2. Extract this project and open Command Prompt or PowerShell in the project folder.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirement.txt
   ```

5. Create a Hugging Face access token with inference permissions:
   https://huggingface.co/settings/tokens

6. Open `.env` and replace `hf_your_token_here` with your token.
7. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Audio input

- **Record Voice:** browser microphone access is provided by `streamlit-mic-recorder`. Allow microphone permissions when prompted. The component's Stop button finalizes the recording.
- **Upload Audio:** WAV, MP3, and M4A.
- **Sample Audio:** add an actual audio file to the `samples` directory. The app does not invent or ship a fake sample transcript/audio.

## Model and API notes

- ASR: `openai/whisper-large-v3` through Hugging Face Inference.
- Chat: `meta-llama/Meta-Llama-3.1-8B-Instruct` through `HuggingFaceEndpoint`.
- LangChain chain: `PromptTemplate | HuggingFaceEndpoint | StrOutputParser()`.

The Hugging Face account must have access to the selected models and a compatible Inference Provider/endpoint. Availability, rate limits, model licensing, and provider support can change. API errors are surfaced in the UI; the app never substitutes mock transcription or fabricated model responses.

If Whisper's hosted inference endpoint does not return a language field, the UI reports the selected language hint or “Not provided by endpoint”; it does not claim an independently verified language detection result.

## Security

- The token is loaded from `.env` (or Streamlit secrets).
- Never publish `.env`, paste your token into source code, or commit it to version control.
- Rotate the token immediately if it is exposed.
- For deployment, use your hosting provider's secret manager rather than uploading `.env`.

## Troubleshooting

- **Token missing:** check the variable name `HUGGINGFACEHUB_API_TOKEN` and restart Streamlit.
- **401/403:** verify the token is valid and has inference permissions/model access.
- **429:** Hugging Face quota or provider rate limit; retry later or configure a dedicated endpoint.
- **Model/provider error:** check current Hugging Face Inference Provider support for both model IDs.
- **Microphone unavailable:** use HTTPS or localhost, grant browser microphone permission, and verify `streamlit-mic-recorder` installed correctly.
- **Audio rejected:** convert the file to a supported format and try again.

## Project structure

```text
ai_voice_to_text_agent/
├── app.py
├── requirement.txt
├── .env
├── README.md
└── samples/
    └── (add your own sample audio here)
```
