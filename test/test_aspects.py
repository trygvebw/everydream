import ldm.data.aspects as aspects

resolutions = [512, 576, 640, 704, 768]
oops = [532, 576, 640, 704, 768]

for res in resolutions:
    example_aspects = aspects.get_aspect_buckets(res)
    print(f" *{res} buckets: {example_aspects}")

    max_pixels = example_aspects[0][0] * example_aspects[0][1]

    for aspect in example_aspects:
        pixels = aspect[0] * aspect[1]
        print (f"max: {max_pixels}: {aspect}: {pixels}, pct {pixels/max_pixels:.2f}")
        assert pixels <= max_pixels, f" * {aspect} is larger than {max_pixels}"