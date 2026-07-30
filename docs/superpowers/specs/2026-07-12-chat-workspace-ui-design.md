# Chat Workspace UI Design

## Goal

Turn the Streamlit agent page from a document-like layout into a GPT-like chat workspace. Users should be able to keep the conversation input visible while message history scrolls inside the chat area, without dragging the whole web page downward to continue talking.

## Scope

This design only changes `scripts/agent_app.py` presentation behavior and related app tests. It does not change the planner, executor, tools, memory schema, or runtime semantics.

## Recommended Approach

Use a fixed-height main chat workspace:

- Keep configuration, login, runtime reset, and thread selection in the Streamlit sidebar.
- Replace the current two-column main document flow with a chat-first page.
- Render the current thread title and runtime status in a compact header.
- Render messages inside a fixed-height scrollable chat panel.
- Keep the chat input visually anchored below the scrollable panel.
- Move plan, step, tool result, blackboard, diagnostics, and confirmation details into compact expandable panels.

## Components

### App Shell

The app shell injects scoped CSS for:

- Full-height main container.
- A centered chat column with a readable maximum width.
- A scrollable message panel using `overflow-y: auto`.
- A compact execution panel that does not dominate the first viewport.

### Chat Workspace

The chat workspace owns:

- Current thread title.
- Last runtime status.
- Chat message rendering.
- Chat input.

Message history remains sourced from `MemoryStore.list_messages(thread.id)`. No message storage behavior changes.

### Execution Details

The execution details panel remains available but collapsed by default. It contains:

- Final answer or current status.
- Human confirmation controls when required.
- Plan steps.
- Tool results.
- Execution digest.
- Evaluation.
- Typed state blackboard.
- Diagnostics.

This keeps developer visibility without making plan/step output the dominant UI.

## Interaction Behavior

- Users can scroll inside the chat history panel.
- The browser page itself should remain mostly stable during normal conversation.
- Example questions still populate the pending question flow.
- Confirmation-required steps remain actionable from the execution panel.
- The default runtime remains Plan-and-Execute; State Graph remains optional.

## Error Handling

If there is no selected thread, show the existing empty-thread guidance. If there is no agent state yet, the execution panel shows a short hint instead of a large blank area.

## Testing

Update app-level tests to cover:

- The UI style injection helper returns fixed chat workspace CSS.
- The chat workspace helper renders through Streamlit primitives without requiring agent execution.
- Existing app helper tests continue to pass.

Manual verification:

- Start `streamlit run scripts/agent_app.py`.
- Log in, select a thread, send a message.
- Confirm chat history scrolls inside the panel and the input remains accessible.
- Expand execution details and confirm plan/tool data is still visible.
