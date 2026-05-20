import os

    st.subheader("📌 Answer")

    st.write(answer)

    # =================================
    # AUDIO
    # =================================

    audio_path = speak(answer)

    st.audio(audio_path)

    # =================================
    # AVATAR VIDEO
    # =================================

    st.subheader("🎬 AI Avatar")

    with st.spinner("Generating AI video..."):

        video = generate_heygen_video(answer)

    # =================================
    # SUCCESS
    # =================================

    if video:

        st.success("Avatar generated ✅")

        st.video(video)

    # =================================
    # FALLBACK
    # =================================

    else:

        st.warning(
            "Avatar API unavailable. Fallback mode activated."
        )

        st.image(
            "https://i.imgur.com/6VBx3io.png",
            width=300
        )

        st.info(
            "Audio explanation is available below."
        )

    # =================================
    # YOUTUBE VIDEOS
    # =================================

    st.subheader("🎥 Arabic Educational Videos")

    try:

        videos = search_youtube(final_question)

        for v in videos:

            st.write(v["title"])

            st.video(v["url"])

    except Exception:

        st.warning(
            "Unable to load YouTube videos"
        )
