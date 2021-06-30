from PIL import Image

colorImage  = Image.open("./3.png")

r = colorImage.rotate(60)
r.save('./img%s-60.png')
r = colorImage.rotate(90)
r.save('./img%s-90.png')
r = colorImage.rotate(120)
r.save('./img%s-120.png')
r = colorImage.rotate(180)
r.save('./img%s-180.png')
r = colorImage.rotate(240)
r.save('./img%s-240.png')
r = colorImage.rotate(300)
r.save('./img%s-300.png')

