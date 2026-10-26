import os

import httpx
import streamlit as st

# Configurable so a future public deployment can point this at a hosted
# backend instead of localhost, with no code changes.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Ask a Primary Source", page_icon="🕯️")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English&family=Cormorant+Garamond:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cormorant Garamond', serif;
    }

    /* Dim, candlelit room — the single background for the whole page */
    .stApp {
        background-color: #2b1b10;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='matrix' values='0 0 0 0 0.9  0 0 0 0 0.65  0 0 0 0 0.35  0 0 0 0.05 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"),
            radial-gradient(circle at 20% 15%, rgba(230, 160, 70, 0.25) 0%, transparent 45%),
            radial-gradient(circle at 80% 85%, rgba(0, 0, 0, 0.4) 0%, transparent 60%),
            linear-gradient(135deg, #3b2614 0%, #2b1b10 60%, #1c1209 100%);
    }

    /* Remove Streamlit's own top toolbar (Deploy button, menu) entirely,
       and the thin colored status bar above it. */
    [data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    /* Light, gold-ish text so it's legible against the dark background */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    label,
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #e8d8ae !important;
    }

    h1, h2, h3 {
        font-family: 'IM Fell English', serif !important;
        color: #e8c27a !important;
        text-shadow: 0 0 8px rgba(230, 160, 70, 0.45);
    }

    h1 {
        border-bottom: 2px solid #c99b4e;
        padding-bottom: 0.3em;
    }

    div[data-testid="stTextInput"] input {
        background-color: #faf3e0;
        border: 1px solid #8b5a2b;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.1em;
        color: #3b2b1a;
    }

    div[data-testid="stButton"] button {
        background-color: #8b5a2b;
        color: #faf3e0;
        border: 1px solid #5a3a1a;
        font-family: 'IM Fell English', serif;
        letter-spacing: 1px;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #6e4620;
        color: #faf3e0;
        border: 1px solid #3b2b1a;
    }

    div[data-testid="stExpander"] {
        background-color: #faf3e0;
        border: 1px solid #c2a36b;
        border-radius: 4px;
    }

    /* Keep source text dark on the light expander panels, overriding the
       page-wide light-text rule above (this selector is more specific, so
       it wins regardless of order). */
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        color: #3b2b1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕯️ Ask a Primary Source")
st.markdown(
    "*Consult the ancient, medieval, and early modern sources — "
    "every answer is drawn and cited from the texts themselves.*"
)

with st.expander("ℹ️ What can I actually ask?"):
    st.markdown(
        "This chatbot only knows what's written in **3 specific historical texts** "
        "— it has no access to general knowledge beyond them:\n\n"
        "- **Tacitus, *Germania*** (~98 AD) — ancient Rome's account of the Germanic tribes\n"
        "- **Bede, *Ecclesiastical History of the English People*** (completed 731 AD) — "
        "early medieval England, up through the early 8th century\n"
        "- **Machiavelli, *The Prince*** (1513) — early modern Florence and Italian politics\n\n"
        "Questions about people, places, or events outside these three texts and eras — "
        "even real, well-known historical ones — won't have an answer, since the system "
        "only draws from what's actually written in these specific books, on purpose."
    )

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Consulting the archives..."):
        try:
            response = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            st.error(f"Couldn't reach the API: {e}")
        else:
            st.subheader("✒️ Answer")
            st.write(data["answer"])

            st.subheader("📜 Sources")
            st.caption(
                "These are the 5 passages the system judged most relevant to your "
                "question — the answer above may only draw on one or two of them. "
                "All are shown for transparency, so you can check the reasoning "
                "yourself rather than take the citation on faith."
            )
            for i, source in enumerate(data["sources"], start=1):
                label = f"Source {i}: {source['source_title']} — {source['chapter'] or 'no chapter listed'}"
                with st.expander(label):
                    st.caption(f"{source['author']} · {source['era']}")
                    st.write(source["text"])
