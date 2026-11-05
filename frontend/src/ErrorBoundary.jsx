import { Component } from "react";

/**
 * Keeps a failure in one view from blanking the whole app. It clears itself when `resetKey`
 * changes, so opening another trace recovers without a reload.
 */
export default class ErrorBoundary extends Component {
  state = { failed: null };

  static getDerivedStateFromError(error) {
    return { failed: error };
  }

  componentDidCatch(error) {
    console.error(`Model Run could not show ${this.props.label}:`, error);
  }

  componentDidUpdate(previous) {
    if (this.state.failed && previous.resetKey !== this.props.resetKey)
      this.setState({ failed: null });
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="view-error" role="alert">
        <strong>Could not show {this.props.label}.</strong>
        <span>
          This capture may be incomplete or from another version. Other views
          and captures still work.
        </span>
        <small>{String(this.state.failed?.message || this.state.failed)}</small>
        {this.props.global ? (
          <button onClick={() => window.location.reload()}>Reload</button>
        ) : (
          <button onClick={() => this.setState({ failed: null })}>
            Try again
          </button>
        )}
      </div>
    );
  }
}
