import HttpBridge from 'react-native-http-bridge';

export const startServer = () => {
  HttpBridge.start(1337, 'localhost', (request) => {
    // Handle requests here
    // For example, you could return a simple HTML page for testing:
    if (request.type === 'GET' && request.url === '/') {
      HttpBridge.respond(request.requestId, 200, 'OK', {'Content-Type': 'text/html'}, '<h1>Hello World</h1>');
    }

    // For streaming video, you'd need to capture the video data and send it in response.
    // This part is complex and depends on how you handle video capture and encoding.
  });
};

