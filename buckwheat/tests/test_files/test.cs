private void MyTaskCompletedCallback(IAsyncResult ar)
{
  // get the original worker delegate and the AsyncOperation instance
  MyTaskWorkerDelegate worker =
    (MyTaskWorkerDelegate)((AsyncResult)ar).AsyncDelegate;
  AsyncOperation async = (AsyncOperation)ar.AsyncState;

  // finish the asynchronous operation
  worker.EndInvoke(ar);

  // clear the running task flag
  lock (_sync)
  {
    _myTaskIsRunning = false;
  }

  // raise the completed event
  AsyncCompletedEventArgs completedArgs = new AsyncCompletedEventArgs(null,
    false, null);
  async.PostOperationCompleted(
    delegate(object e) { OnMyTaskCompleted((AsyncCompletedEventArgs)e); },
    completedArgs);
}
