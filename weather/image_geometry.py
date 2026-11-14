"""Aspect-preserving geometry diagnostic; the formal JMDA default is unchanged."""

from PIL import Image


class LetterboxSquare:
    def __init__(self, size, fill):
        if size < 1:
            raise ValueError("Positive canvas size required")
        self.size, self.fill = int(size), fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size/max(width, height)
        resized = image.resize((max(1, round(width*scale)), max(1, round(height*scale))), Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size-resized.width)//2, (self.size-resized.height)//2))
        return canvas

    def __repr__(self):
        return f"LetterboxSquare(size={self.size}, fill={self.fill})"
