function disableBackButton() {
  window.history.forward();
}
setTimeout("disableBackButton()", -1);