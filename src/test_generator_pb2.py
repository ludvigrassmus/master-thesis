# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: test_generator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='test_generator.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x14test_generator.proto\"\x14\n\x04Word\x12\x0c\n\x04text\x18\x01 \x01(\t\"\x18\n\x08Sentence\x12\x0c\n\x04text\x18\x01 \x01(\t22\n\rTestGenerator\x12!\n\x0bGetSentence\x12\x05.Word\x1a\t.Sentence\"\x00\x62\x06proto3'
)




_WORD = _descriptor.Descriptor(
  name='Word',
  full_name='Word',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='Word.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=44,
)


_SENTENCE = _descriptor.Descriptor(
  name='Sentence',
  full_name='Sentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='Sentence.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=70,
)

DESCRIPTOR.message_types_by_name['Word'] = _WORD
DESCRIPTOR.message_types_by_name['Sentence'] = _SENTENCE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Word = _reflection.GeneratedProtocolMessageType('Word', (_message.Message,), {
  'DESCRIPTOR' : _WORD,
  '__module__' : 'test_generator_pb2'
  # @@protoc_insertion_point(class_scope:Word)
  })
_sym_db.RegisterMessage(Word)

Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), {
  'DESCRIPTOR' : _SENTENCE,
  '__module__' : 'test_generator_pb2'
  # @@protoc_insertion_point(class_scope:Sentence)
  })
_sym_db.RegisterMessage(Sentence)



_TESTGENERATOR = _descriptor.ServiceDescriptor(
  name='TestGenerator',
  full_name='TestGenerator',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=72,
  serialized_end=122,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetSentence',
    full_name='TestGenerator.GetSentence',
    index=0,
    containing_service=None,
    input_type=_WORD,
    output_type=_SENTENCE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TESTGENERATOR)

DESCRIPTOR.services_by_name['TestGenerator'] = _TESTGENERATOR

# @@protoc_insertion_point(module_scope)
