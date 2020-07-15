// Rectangle instance declaration
function Rectangle(height, width) {
  this.height = height
  this.width = width
}

/* Calculating area of Rectangle.
Create function clacArea.
Area calculations = height * width.
 */
Rectangle.prototype.calcArea = function calcArea() {
  return this.height * this.width
}
