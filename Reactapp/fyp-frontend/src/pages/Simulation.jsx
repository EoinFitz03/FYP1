import StreamPanel from "../components/StreamPanel";

export default function Simulation() {
  return (
    <StreamPanel
      title="Simulation"
      timed={true}
      durationSec={15}
      doorSim={true}
      openGesture="thumbs_up"
      closeGesture="open_palm"
    />
  );
}