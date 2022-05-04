import schemdraw
import schemdraw.elements as elm

d = schemdraw.Drawing()
d += elm.Resistor().label('100KΩ')
d += elm.Capacitor().down().label('0.1μF', loc='bottom')
d += elm.Line().left()
d += elm.Ground()
d += elm.SourceV().up().label('10V')
d.draw()