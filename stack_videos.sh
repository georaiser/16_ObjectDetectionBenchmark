#!/bin/bash

# Leer nombres de videos en un array
mapfile -t videos < videos.txt
count=${#videos[@]}

# Validar cantidad de videos
if [[ $count -ne 4 && $count -ne 16 ]]; then
    echo "Error: Solo se permiten 4 o 16 videos."
    exit 1
fi

# Preparar opciones de entrada y filter_complex
input_options=""
filter_complex=""

for i in "${!videos[@]}"; do
    input_options+="-ss 45 -i ${videos[i]} "
    filter_complex+="[$i:v]scale=480:270[v$i];"
done

if [[ $count -eq 4 ]]; then
    # Stack 2x2
    filter_complex+="[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[out]"
elif [[ $count -eq 16 ]]; then
    # Stack 4x4
    filter_complex+="[v0][v1][v2][v3]hstack=4[row0];[v4][v5][v6][v7]hstack=4[row1];"
    filter_complex+="[v8][v9][v10][v11]hstack=4[row2];[v12][v13][v14][v15]hstack=4[row3];"
    filter_complex+="[row0][row1][row2][row3]vstack=4[out]"
fi

# Ejecutar FFmpeg
ffmpeg $input_options -filter_complex "$filter_complex" -map "[out]" -c:v libx264 -crf 23 -preset fast output.mp4
