
class BroadcastTraffic:
	traffic = []
	nr_packets = 0
	nr_data_packets = 0
	nr_sack_packets = 0
	nr_join_req = 0
	nr_join_acp = 0

	@classmethod
	def __inc_count(cls, packet):
		from communications import DataPacket, SackPacket
		cls.nr_packets += 1
		if isinstance(packet, DataPacket):
			cls.nr_data_packets += 1
		if isinstance(packet, SackPacket):
			cls.nr_sack_packets += 1
		# if isinstance(packet, JoinRequest):
		# 	cls.nr_join_req += 1
		# if isinstance(packet, JoinAccept):
		# 	cls.nr_join_acp += 1

	@classmethod
	def __dec_count(cls, packet):
		from communications import DataPacket, SackPacket
		cls.nr_packets -= 1
		if isinstance(packet, DataPacket):
			cls.nr_data_packets -= 1
		if isinstance(packet, SackPacket):
			cls.nr_sack_packets -= 1
		# if isinstance(packet, JoinRequest):
		# 	cls.nr_join_req -= 1
		# if isinstance(packet, JoinAccept):
		# 	cls.nr_join_acp -= 1

	@classmethod
	def add_generator(cls, env, packet):
		cls.traffic.append(packet)
		cls.__inc_count(packet)
		yield env.timeout(packet.rec_time)
		packet.sent = True
		cls.__dec_count(packet)
		cls.traffic.remove(packet)

	@classmethod
	def add_and_wait(cls, env, packet):
		return env.process(cls.add_generator(env, packet))

	@classmethod
	def is_p_cls_broadcasting(cls, packet_class):
		from communications import DataPacket, SackPacket
		if packet_class == DataPacket:
			return cls.nr_data_packets > 0
		if packet_class == SackPacket:
			return cls.nr_sack_packets > 0
		# if packet_class == JoinRequest:
		# 	return cls.nr_join_req > 0
		# if packet_class == JoinAccept:
		# 	return cls.nr_join_acp > 0
		return False
