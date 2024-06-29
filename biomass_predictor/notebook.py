from pickle import load

import gradio as gr
import pandas as pd

with open("model.pkl", "rb") as f:
    reg = load(f)

with open("MODEL.md", "r") as f:
    docs = f.read()


def get_dry_matter_consumption(
    number_of_days,
    number_of_cows,
    average_cattle_weight,
    average_daily_cattle_weight_gain,
    consumption_rate,
):
    consumption_per_cow = consumption_rate / 100 * average_cattle_weight + (
        0.1 * average_daily_cattle_weight_gain
    )
    result = number_of_cows * number_of_days * consumption_per_cow
    return result


def update(
    current_biomass,
    cumulative_temperature,
    cumulative_humidity,
    cumulative_rain,
    cumulative_light,
    number_of_days,
    number_of_cows,
    average_cattle_weight,
    average_daily_cattle_weight_gain,
    consumption_rate,
    pasture_rate,
    plot_size,
):
    df = pd.DataFrame(
        [
            {
                "previous_biomass": current_biomass,
                "cumulative_temperature": cumulative_temperature,
                "cumulative_humidity": cumulative_humidity,
                "cumulative_rain": cumulative_rain,
                "cumulative_light": cumulative_light,
            }
        ]
    )

    predicted_biomass_growth = reg.predict(df)[0]
    predicted_dry_matter_consumption = get_dry_matter_consumption(
        number_of_days,
        number_of_cows,
        average_cattle_weight,
        average_daily_cattle_weight_gain,
        consumption_rate,
    )
    predicted_total_biomass = (
        predicted_biomass_growth * plot_size * pasture_rate / 100
        - predicted_dry_matter_consumption
    )

    predicted_biomass_level = predicted_total_biomass / plot_size
    return predicted_biomass_level


def toggle_biomass_level_warning(biomass_level):
    if float(biomass_level) <= 350:
        return "Nivel de biomasa estimado <= 350 kg/Ha, se recomienda rotar el ganado."
    else:
        return ""


css = """
#warning {background-color: #e06666}
"""

with gr.Blocks(css=css) as demo:
    with gr.Accordion("Ver documentación", open=False):
        gr.Markdown(docs)

    with gr.Row():
        with gr.Column():
            current_biomass = gr.Number(label="Biomasa actual (kg/Ha)", value=2500)
            cumulative_temperature = gr.Number(
                label="Temperatura cumulativa (°C)", value=120
            )
            cumulative_humidity = gr.Number(label="Humedad cumulativa (%)", value=350)
            cumulative_rain = gr.Number(label="Lluvia cumulativa (mm)", value=200)
            cumulative_light = gr.Number(label="Luz cumulativa (lux)", value=85000)
        with gr.Column():
            number_of_days = gr.Number(label="Cantidad de días", value=5)
            number_of_cows = gr.Number(label="Cabezas de ganado", value=100)
            average_cattle_weight = gr.Number(label="Peso por cabeza (kg)", value=267)
            average_daily_cattle_weight_gain = gr.Number(
                label="Incremento de peso diario por cabeza (kg)", value=0.3
            )
            consumption_rate = gr.Number(
                label="Consumo diario por cabeza (% de peso vivo)", value=2.51
            )
            pasture_rate = gr.Number(
                label="Porcentaje de pastura del potrero (%)", value=90
            )
            plot_size = gr.Number(label="Superficie del potrero (Ha)", value=10)
        with gr.Column():
            predicted_biomass = gr.Number(
                label="Biomasa estimada (kg/Ha)", interactive=False
            )
            low_biomass_level_warning = gr.Markdown("", elem_id="warning")
            predicted_biomass.change(
                toggle_biomass_level_warning,
                predicted_biomass,
                low_biomass_level_warning,
            )

    btn = gr.Button("Predecir")
    btn.click(
        fn=update,
        inputs=[
            current_biomass,
            cumulative_temperature,
            cumulative_humidity,
            cumulative_rain,
            cumulative_light,
            number_of_days,
            number_of_cows,
            average_cattle_weight,
            average_daily_cattle_weight_gain,
            consumption_rate,
            pasture_rate,
            plot_size,
        ],
        outputs=predicted_biomass,
    )


if __name__ == "__main__":
    demo.launch()
