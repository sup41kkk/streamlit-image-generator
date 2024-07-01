import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select

# UI configurations
st.set_page_config(page_title="Генератор Изображений",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Студия художественного перевода текстов в изображения]")

# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Resources text, link, and logo
replicate_text = "Модель Stability AI SDXL на Replicate"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Привет! Начни здесь ↓**\nДля корректной работы приложения пишите свои запросы на английском языке", icon="👋🏾")
            with st.expander(":rainbow[**Настройте вывод здесь**]"):
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Ширина выходного изображения", value=1024)
                height = st.number_input("Высота выходного изображения", value=1024)
                num_outputs = st.slider(
                    "Количество изображений для вывода", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Выбери модель для', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                         'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Количество шагов денойзинга", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Масштаб для безклассификационного управления", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Сила подсказки при использовании img2img/inpaint (1.0 соответствует полному разрушению информации в изображении)", value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Выберите стиль уточнения (исключены другие 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Доля шума для `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
            prompt = st.text_area(
                ":orange[**Введите запрос: начните печатать, Шекспир ✍🏾**]",
                value="An astronaut riding a rainbow unicorn, cinematic, dramatic")
            negative_prompt = st.text_area(":orange[**Что не хотите видеть в изображении? 🙅🏽‍♂️**]",
                                           value="the absolute worst quality, distorted features",
                                           help="Это негативный запрос, введите то, что не хотите видеть в сгенерированном изображении")

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Отправить", type="primary", use_container_width=True)

        # Credits and resources
        st.divider()
        st.markdown(
            ":orange[**Ресурсы:**]  \n"
            f"<img src='{replicate_logo}' style='height: 1em'> [{replicate_text}]({replicate_link})",
            unsafe_allow_html=True
        )

        return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Преобразуем ваши слова в искусство...', expanded=True) as status:
            st.write("⚙️ Модель запущена")
            st.write("🙆‍♀️ Поднимитесь и потянитесь, пока идет процесс")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "num_outputs": num_outputs,
                                "scheduler": scheduler,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "prompt_stregth": prompt_strength,
                                "refine": refine,
                                "high_noise_frac": high_noise_frac
                            }
                        )
                        if output:
                            st.toast(
                                'Ваше изображение создано!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Сгенерированное изображение 🎈",
                                             use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Не удалось получить изображение {i+1} от {image}. Код ошибки: {response.status_code}", icon="🚨")
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Скачать все изображения**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Изображения сгенерированы!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Произошла ошибка: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    with gallery_placeholder.container():
        img = image_select(
            label="Понравилось? Нажмите правой кнопкой мыши и сохраните! 😉",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
                "gallery/cheetah.png", "gallery/viking.png",
            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
                      "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
                      ],
            use_container_width=True
        )


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    main_page(submitted, width, height, num_outputs, scheduler, num_inference_steps,
              guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt)


if __name__ == "__main__":
    main()
