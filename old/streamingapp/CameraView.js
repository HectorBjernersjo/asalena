import React, { useState, useEffect, useRef } from 'react';
import { Camera } from 'expo-camera';
import { View, Text } from 'react-native';
import { startServer } from './HttpServer';

const CameraView = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
    startServer(); // Start the HTTP server
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <Camera style={{ flex: 1 }} ref={cameraRef} type={Camera.Constants.Type.back}>
      {/* Camera content */}
    </Camera>
  );
};

export default CameraView;

