import streamlit as st
from langchain_ai import extract_laundry_info, explain_result
from fuzzy_logic import recommend_wash

st.set_page_config(page_title="Smart Laundry Advisor", page_icon="🧺", layout="centered")

st.title("🧺 AI-Based Smart Laundry Wash Setting Advisor")
st.caption("LangChain + LLM + Fuzzy Logic Mini Project")

st.write(
    "Describe your laundry in normal language. The AI extracts useful information, "
    "then the fuzzy inference system recommends a wash setting."
)

user_text = st.text_area(
    "Describe your laundry:",
    placeholder="Example: I have about 5 kg of cotton clothes. They are very dirty and I want to save water.",
    height=130,
)

if st.button("🔍 Analyze Laundry", type="primary"):
    if not user_text.strip():
        st.warning("Please describe your laundry first.")
        st.stop()

    with st.spinner("AI is understanding your description..."):
        try:
            info = extract_laundry_info(user_text)
        except Exception as e:
            st.error(f"AI extraction failed: {e}")
            st.stop()

    st.subheader("🤖 AI Understanding")
    c1, c2 = st.columns(2)
    c1.metric("Load", f"{info['load_kg']} kg")
    c2.metric("Dirt Level", f"{info['dirt_level']}/100")
    c3, c4 = st.columns(2)
    c3.metric("Water-Saving Priority", f"{info['water_saving']}/100")
    c4.write(f"**Fabric:** {info['fabric']}")

    result = recommend_wash(
        load_kg=info["load_kg"],
        dirt_level=info["dirt_level"],
        water_saving=info["water_saving"],
    )

    st.subheader("🧠 Fuzzy Logic Result")
    st.success(f"### Recommended Wash: {result['wash_intensity']}")

    a, b, c = st.columns(3)
    a.metric("Wash Time", f"{result['wash_time']} min")
    b.metric("Water Level", result["water_level"])
    c.metric("Spin", result["spin"])

    with st.expander("🔬 Fuzzy-system details"):
        st.write("**Membership values**")
        st.json(result["memberships"])
        st.write("**Rule activations**")
        st.json(result["rule_activations"])
        st.write(f"**Defuzzified wash intensity:** {result['crisp_intensity']:.2f}/100")

    with st.spinner("Generating a conversational explanation..."):
        try:
            explanation = explain_result(user_text, info, result)
            st.subheader("💡 AI Explanation")
            st.write(explanation)
        except Exception as e:
            st.info(
                "The fuzzy result is ready. Add your LLM API key to enable the "
                f"conversational explanation. ({e})"
            )

st.divider()
st.caption("Academic mini project — demonstration system. It is not a real washing-machine control system.")
